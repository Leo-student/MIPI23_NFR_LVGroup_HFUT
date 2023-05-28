import time
import torch
import random

import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from skimage import io


from datetime import datetime
from utils import *
import synthesis
from options import TestOptions
# from condition_model import NAFNet
from model.nbn_model import UNetD
from losses import LossCont, LossFreqReco, LossGan, LossCycleGan, LossPerceptual
from datasets import Flare_Image_Loader, SingleImgDataset

from skimage.metrics import mean_squared_error
from statistics import mean
from tqdm import tqdm

from log import Log
log = Log(__name__).getlog()


log.info('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')

opt = TestOptions().parse()
image_dir = opt.outputs_dir + '/' + opt.experiment + '/infer'
clean_dir(image_dir, delete=False)


models_dir, log_dir, train_images_dir, val_images_dir, test_images_dir = prepare_dir(opt.outputs_dir, opt.experiment, delete=False)


log.info('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')

infer_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))

log.info('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = UNetD(3)
print_para_num(model)
device = torch.device('cuda:0')
model = model.to(device)

    
#log.info('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')

#log.info('---------------------------------------- step 5/5 : training... ----------------------------------------------------')


def generate(model, models_dir, test_images_dir ):
   
   
    state = torch.load(opt.model_path)
    # state = torch.load(models_dir  + '/best.pth')
    print(state['epoch'])
    new_state_dict = {}
    for key, value in state['model'].items():
        if key.startswith('module.'):  # 如果键以 "module." 开头，则去除该前缀
            new_key = key[len('module.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    # model = model.module ###
    model.load_state_dict(new_state_dict)
    epoch = state['epoch']
   
    # print('successfully loading pretrained model.')
    model.eval()

    time_meter = AverageMeter()
    for i, (img, path) in tqdm(enumerate(infer_dataloader, 1), colour = "YELLOW", leave=False, total=len(infer_dataloader), ncols=70):
    # for i, (img, path) in enumerate(infer_dataloader):
        img = img.cuda()

        with torch.no_grad():
            start_time = time.time()
            pred = model(img)
            pred_blend = synthesis.blend_light_source(img.cuda(), pred.cuda())
            times = time.time() - start_time

        # pred_clip = torch.clamp(pred_blend, 0, 1)
        pred_clip = torch.clamp(pred, 0, 1)

        time_meter.update(times, 1)

        # print('Iteration: {:0>3}/{} Processing image...Path {} Time {:.3f}'.format((i+1) ,len(infer_dataset),path, times))


        if opt.save_image:
            save_image(pred_clip, test_images_dir + '/' + os.path.basename(path[0]))

    # print('Avg time: {:.3f}'.format(time_meter.average()))
    return epoch

def   evaluate(test_images_dir, epoch):
    # input_dir =
    # output_dir =
    input_folder = os.path.join(test_images_dir)
    gt_folder = os.path.join(opt.data_source, 'val/gt')
    mask_folder = os.path.join(opt.data_source, 'val/mask')

    output_filename = os.path.join(test_images_dir, 'scores.txt')

    mask_type_list=['glare','streak','global']
    gt_list=sorted(os.listdir(gt_folder))
    input_list = list(map(lambda x: os.path.join(input_folder,x.replace('gt', 'input')), gt_list))
    mask_list = list(map(lambda x: os.path.join(mask_folder,x.replace('gt', 'mask')), gt_list))
    gt_list = list(map(lambda x: os.path.join(gt_folder,x), gt_list))

    img_num=len(gt_list)
    metric_dict={'glare':[],'streak':[],'global':[]}

    def extract_mask(img_seg):
    # Return a dict with 3 masks including streak,glare,global(whole image w/o light source), masks are returned in 3ch.
    # glare: [255,255,0]
    # streak: [255,0,0]
    # light source: [0,0,255]
    # others: [0,0,0]
        mask_dict={}
        streak_mask=(img_seg[:,:,0]-img_seg[:,:,1])/255
        glare_mask=(img_seg[:,:,1])/255
        global_mask=(255-img_seg[:,:,2])/255
        mask_dict['glare']=[np.sum(glare_mask)/(512*512),np.expand_dims(glare_mask,2).repeat(3,axis=2)] #area, mask
        mask_dict['streak']=[np.sum(streak_mask)/(512*512),np.expand_dims(streak_mask,2).repeat(3,axis=2)]
        mask_dict['global']=[np.sum(global_mask)/(512*512),np.expand_dims(global_mask,2).repeat(3,axis=2)]
        return mask_dict

    for i in range(img_num):
        img_gt=io.imread(gt_list[i])
        img_input=io.imread(input_list[i])
        img_seg=io.imread(mask_list[i])
        for mask_type in mask_type_list:
            mask_area,img_mask=extract_mask(img_seg)[mask_type]
            if mask_area>0:
                img_gt_masked=img_gt*img_mask
                img_input_masked=img_input*img_mask
                input_mse=mean_squared_error(img_gt_masked, img_input_masked)/(255*255*mask_area)
                input_psnr=10 * np.log10((1.0 ** 2) / input_mse)
                metric_dict[mask_type].append(input_psnr)

    glare_psnr=mean(metric_dict['glare'])
    streak_psnr=mean(metric_dict['streak'])
    global_psnr=mean(metric_dict['global'])

    mean_psnr=mean([glare_psnr,streak_psnr,global_psnr])
   

    with open(output_filename, 'w') as f:
        f.write('{}: {:.3f}\n'.format('G-PSNR', glare_psnr))
        f.write('{}: {:.3f}\n'.format('S-PSNR', streak_psnr))
        f.write('{}: {:.3f}\n'.format('ALL-PSNR', global_psnr))
        f.write('{}: {:.3f}\n'.format('Score', mean_psnr))
       

    
    log.info('{}: {:.3f} dB. {}: {:.3f} dB. {}: {:.3f} dB. Score: {:.3f}  '.format('G-PSNR', glare_psnr, 'S-PSNR', streak_psnr, 'ALL-PSNR', global_psnr, mean_psnr))
    

def main(opt):

    
    epoch = generate(model, models_dir,test_images_dir)
    evaluate(test_images_dir, epoch)

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    
    main(opt)


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
from options import TrainOptions
from condition_model import NAFNet
from losses import LossCont, LossFreqReco, LossGan, LossCycleGan, LossPerceptual
from datasets import Flare_Image_Loader, SingleImgDataset

from skimage.metrics import mean_squared_error
from statistics import mean
from tqdm import tqdm

from log import Log
log = Log(__name__).getlog()
# from infer import evaluate, generate

# Tensor type

#log.info('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')

#print all options
opt = TrainOptions().parse()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '') 
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

models_dir, log_dir, train_images_dir, val_images_dir, test_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(opt.resume))

set_random_seed(opt.seed)

#define tensorboard 
writer = SummaryWriter(log_dir=log_dir)

#log.info('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
#log.info('training data loading...')
if opt.debug :
    train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/2try', crop=opt.crop)
else :
    train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
# train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
train_dataset.load_scattering_flare()

train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True, prefetch_factor = 4)
#log.info('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

#log.info('validating data loading...')
val_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
#log.info('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))



#log.info('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = NAFNet()
print_para_num(model)
num_gpus = torch.cuda.device_count()
device_ids = [int(i) for i in range(len(cuda_visible_devices.split(','))) ]
if opt.data_parallel:
    
    #log.info('\t----------------------------------------data_parallel----------------------------------------------')

    
    # #log.info(f'Number of available GPUs is {num_gpus} \n They are"{cuda_visible_devices}" at {device_ids}.')
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
if not opt.data_parallel:
    device = torch.device('cuda:0')
    model = model.to(device)
#log.info(f'\tNumber of available GPUs is {num_gpus} They are"{cuda_visible_devices}" at {device_ids}.')
    
#log.info('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_fft = LossFreqReco()
criterion_g = LossGan()

criterion_c = LossCycleGan()


optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [110,140], 0.9)

#log.info('---------------------------------------- step 5/5 : training... ----------------------------------------------------')

best_score = 0
best_epoch = 0
best_psnr = 0





def train(epoch):
    
    model.train()


    max_iter = len(train_dataloader)

    psnr_meter = AverageMeter()
    iter_cont_meter = AverageMeter()
    iter_fft_meter = AverageMeter()
   
    iter_re_meter = AverageMeter()

    iter_timer = Timer()
    # Calculate output of image discriminator (PatchGAN)

    with tqdm(total= len(train_dataloader), colour = "MAGENTA", leave=False, ncols=120 ) as pbar:
        for i, (gts, flares, imgs, _) in enumerate(train_dataloader):
            pbar.set_description('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] '.format(epoch, opt.n_epochs, i + 1, max_iter))
        # for i, (gts, flares, imgs, _) in enumerate(train_dataloader):
            gts, flares, imgs = gts.cuda(), flares.cuda(), imgs.cuda()
            cur_batch = imgs.shape[0]

            
            # ------------------
            #  Train Generators
            # ------------------
            optimizer.zero_grad()
            preds_flare, preds = model(imgs)

            loss_cont =   opt.lambda_flare * criterion_cont(preds_flare, flares) +criterion_cont(preds, gts)
           
            loss_fft =   opt.lambda_flare * criterion_fft(preds_flare, flares) + criterion_cont(preds, gts)
            
            light_source = synthesis.get_highlight_mask(flares)
            mask_gt = synthesis.flare_to_mask(flares)
            mask_lf =  -  light_source + mask_gt
            masked_lf_scene = (1 - mask_lf) * imgs + mask_lf * preds
            masked_lf_flare = (1 - mask_lf) * imgs + mask_lf * preds_flare
            
            
            loss_region = criterion_cont(masked_lf_scene, gts) 
            loss_flare = criterion_fft(masked_lf_flare, gts) 
            
            lambda_region =   (512 * 512 * cur_batch)  / torch.sum(mask_lf).cpu().numpy() 
            # print(lambda_region, torch.sum(mask_gt), torch.sum(mask_lf), torch.sum(light_source) );print(" ")
            
            
            
            # loss =  loss_cont +  lambda_region * loss_region 
            loss =  loss_cont + opt.lambda_fft * loss_fft + lambda_region * loss_region 
            
            loss.backward()
            optimizer.step()


            iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
            iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)


            # if lambda_region >= 1.0:
                # save_image(torch.cat((imgs,preds.detach(),preds_flare.detach(),flares,gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)

            if (i+1) % opt.print_gap == 0:
                writer.add_scalar('Loss_cont', iter_cont_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
                writer.add_scalar('Loss_fft', iter_fft_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
               
            pbar.update(1)

    writer.add_scalar('lr', scheduler.get_last_lr()[0], epoch)
    
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}, models_dir + '/latest.pth')
    scheduler.step()

# def val(epoch):
#     model.eval()

#     # print(''); 
#     #log.info('Validating...')

#     timer = Timer()

#     for i, (img, path) in enumerate(val_dataloader):
#         img = img.cuda()

#         with torch.no_grad():
#             pred_flare, pred = model(img)
#         pred_clip = torch.clamp(pred, 0, 1)
#         pred_flare_clip = torch.clamp(pred_flare, 0, 1)

#         # if i < 5:
#         #     # save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_'.format(epoch) + os.path.basename(path[0]))
#         #     save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_img_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
#         #     save_image(pred_flare_clip, val_images_dir + '/epoch_{:0>4}_flare_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
#         # else:
#         #     break

#     # torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),  'epoch': epoch}, models_dir + '/epoch_{:0>4}.pth'.format(epoch))

#     #log.info('Epoch[{:0>4}/{:0>4}] Time: {:.4f}'.format(epoch, opt.n_epochs, timer.timeit()))#; print('')


def generate(model, models_dir, test_images_dir ):
   
    infer_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
    # print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))
    state = torch.load(models_dir  + '/best.pth')
    # model = model.module ###
    model.load_state_dict(state['model'])
    # model.load_state_dict(torch.load(models_dir  + '/epoch_{:0>4}.pth'.format(epoch)))


    # print('successfully loading pretrained model.')
    model.eval()

    time_meter = AverageMeter()
    for i, (img, path) in tqdm(enumerate(infer_dataloader, 1), colour = "YELLOW", leave=False, total=len(infer_dataloader), ncols=70):
    # for i, (img, path) in enumerate(infer_dataloader):
        img = img.cuda()

        with torch.no_grad():
            start_time = time.time()
            _, pred = model(img)
            pred_blend = synthesis.blend_light_source(img.cpu(), pred.cpu())
            times = time.time() - start_time

        # pred_clip = torch.clamp(pred_blend, 0, 1)
        pred_clip = torch.clamp(pred, 0, 1)

        time_meter.update(times, 1)

        # print('Iteration: {:0>3}/{} Processing image...Path {} Time {:.3f}'.format((i+1) ,len(infer_dataset),path, times))


        if opt.save_image:
            save_image(pred_clip, test_images_dir + '/' + os.path.basename(path[0]))

    # print('Avg time: {:.3f}'.format(time_meter.average()))
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
    global best_score, best_epoch, best_psnr
    if best_score < mean_psnr :
        best_score = mean_psnr
        best_epoch = epoch
        best_psnr = global_psnr
        update_pth_f = 1 
    else :
        best_epoch = best_epoch
        best_psnr = best_psnr
        update_pth_f = 0

    with open(output_filename, 'w') as f:
        f.write('{}: {:.3f}\n'.format('G-PSNR', glare_psnr))
        f.write('{}: {:.3f}\n'.format('S-PSNR', streak_psnr))
        f.write('{}: {:.3f}\n'.format('ALL-PSNR', global_psnr))
        f.write('{}: {:.3f}\n'.format('Score', mean_psnr))
        f.write('{}: {:.3f} {}\n'.format('best', best_score, best_epoch, best_psnr))
        # f.write('DEVICE: CPU\n')
    writer.add_scalar('G-PSNR', glare_psnr, epoch)   
    writer.add_scalar('S-PSNR', streak_psnr, epoch)   
    writer.add_scalar('ALL-PSNR',global_psnr, epoch)   
    writer.add_scalar('mean_psnr',mean_psnr, epoch) 
    writer.add_scalar('best_psnr',best_psnr, epoch) 
    
    log.info('{}: {:.3f} dB. {}: {:.3f} dB. {}: {:.3f} dB. Score: {:.3f} Best{:.3f} at Epoch: {} best pnsr= {:.3f}'.format('G-PSNR', glare_psnr, 'S-PSNR', streak_psnr, 'ALL-PSNR', global_psnr, mean_psnr, best_score,best_epoch,best_psnr))
    return update_pth_f

def main(opt):

    # start_epoch = 1
    # if opt.resume:
    #     state = torch.load(models_dir + '/best.pth')
    #     model.load_state_dict(state["model"])
    #     # optimizer.load_state_dict(state['optimizer'])
    #     # scheduler.load_state_dict(state['scheduler'])

    #     start_epoch = state['epoch'] + 1
    #     #log.info('Resume model from epoch %d' % (start_epoch))

       

    #     # start_epoch = state['epoch'] + 1
    #     # print('Resume d from epoch %d' % (start_epoch))


    # update_best = 0
    # for epoch in range(start_epoch, opt.n_epochs + 1):
    #     train(epoch)

    #     if (epoch) % opt.val_gap == 0:
    #         # val(epoch)
    #         generate(model, models_dir,test_images_dir)
    #         update_best = evaluate(test_images_dir, epoch)
    #         if update_best:
    #             os.rename(models_dir + "/latest.pth", models_dir + "/best.pth")
    #             update_best = 0

    # writer.close()
    generate(model, models_dir,test_images_dir)
    update_best = evaluate(test_images_dir, 126)
if __name__ == '__main__':
    
    torch.manual_seed(opt.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    
    
    main(opt)


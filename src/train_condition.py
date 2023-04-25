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

from utils import *
import synthesis
from options import TrainOptions
from condition_model import NAFNet, Discriminator
from losses import LossCont, LossFreqReco, LossGan, LossCycleGan, LossPerceptual
from datasets import Flare_Image_Loader, SingleImgDataset

from skimage.metrics import mean_squared_error
from statistics import mean
from tqdm import tqdm
# from infer import evaluate, generate

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

# device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")



print('---------------------------------------- step 1/5 : parameters preparing... ----------------------------------------')
opt = TrainOptions().parse()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_idx
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '') 
cuda = True if torch.cuda.is_available() else False
# Tensor type
print(cuda)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
set_random_seed(opt.seed)

models_dir, log_dir, train_images_dir, val_images_dir, test_images_dir = prepare_dir(opt.results_dir, opt.experiment, delete=(not opt.resume))

writer = SummaryWriter(log_dir=log_dir)

print('---------------------------------------- step 2/5 : data loading... ------------------------------------------------')
print('training data loading...')
if opt.debug :
    train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/2try', crop=opt.crop)
else :
    train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
# train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
train_dataset.load_scattering_flare()
print('batch size')
print(opt.train_bs)
train_dataloader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(train_dataset),opt.train_bs))

print('validating data loading...')
val_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
val_dataloader = DataLoader(val_dataset, batch_size=opt.val_bs, shuffle=False, num_workers=opt.num_workers, pin_memory=True)
print('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(val_dataset),opt.val_bs))



print('---------------------------------------- step 3/5 : model defining... ----------------------------------------------')
model = NAFNet()
print_para_num(model)
num_gpus = torch.cuda.device_count()
device_ids = [int(i) for i in range(len(cuda_visible_devices.split(','))) ]
if opt.data_parallel:
    
    print('\t----------------------------------------data_parallel----------------------------------------------')

    
    # print(f'Number of available GPUs is {num_gpus} \n They are"{cuda_visible_devices}" at {device_ids}.')
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    
if not opt.data_parallel:
    device = torch.device('cuda:0')
    model = model.to(device)
print(f'\tNumber of available GPUs is {num_gpus} \n They are"{cuda_visible_devices}" at {device_ids}.')
    
print('---------------------------------------- step 4/5 : requisites defining... -----------------------------------------')
criterion_cont = LossCont()
criterion_fft = LossFreqReco()
criterion_g = LossGan()

criterion_c = LossCycleGan()


optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))


print('---------------------------------------- step 5/5 : training... ----------------------------------------------------')

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
           
            loss =  loss_cont
           
            loss.backward()
            optimizer.step()


            iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
            iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)


            if i == 0:
                save_image(torch.cat((imgs,preds.detach(),preds_flare.detach(),flares,gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)

            if (i+1) % opt.print_gap == 0:
                writer.add_scalar('Loss_cont', iter_cont_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
                writer.add_scalar('Loss_fft', iter_fft_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
               
            pbar.update(1)

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),  'epoch': epoch}, models_dir + '/latest.pth')
    

def val(epoch):
    model.eval()

    print(''); print('Validating...', end=' ')

    timer = Timer()

    for i, (img, path) in enumerate(val_dataloader):
        img = img.cuda()

        with torch.no_grad():
            pred_flare, pred = model(img)
        pred_clip = torch.clamp(pred, 0, 1)
        pred_flare_clip = torch.clamp(pred_flare, 0, 1)

        if i < 5:
            # save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_'.format(epoch) + os.path.basename(path[0]))
            save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_img_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
            save_image(pred_flare_clip, val_images_dir + '/epoch_{:0>4}_flare_'.format(epoch) + os.path.basename(path[0]), nrow=opt.val_bs//2, normalize=True, scale_each=True)
        else:
            break

    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),  'epoch': epoch}, models_dir + '/epoch_{:0>4}.pth'.format(epoch))

    print('Epoch[{:0>4}/{:0>4}] Time: {:.4f}'.format(epoch, opt.n_epochs, timer.timeit())); print('')


def generate(model, epoch, models_dir, test_images_dir ):
   
    infer_dataset = SingleImgDataset(data_source=opt.data_source + '/val')
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    # print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))
    state = torch.load(models_dir  + '/epoch_{:0>4}.pth'.format(epoch))
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

        pred_clip = torch.clamp(pred_blend, 0, 1)

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
    else :
        best_epoch = best_epoch
        best_psnr = best_psnr

    with open(output_filename, 'w') as f:
        f.write('{}: {:.3f}\n'.format('G-PSNR', glare_psnr))
        f.write('{}: {:.3f}\n'.format('S-PSNR', streak_psnr))
        f.write('{}: {:.3f}\n'.format('ALL-PSNR', global_psnr))
        f.write('{}: {:.3f}\n'.format('Score', mean_psnr))
        f.write('{}: {:.3f} {}\n'.format('best', best_score, best_epoch))
        # f.write('DEVICE: CPU\n')
    writer.add_scalar('G-PSNR', glare_psnr, epoch)   
    writer.add_scalar('S-PSNR', streak_psnr, epoch)   
    writer.add_scalar('ALL-PSNR',global_psnr, epoch)   
    writer.add_scalar('mean_psnr',mean_psnr, epoch) 
    print('{}: {:.3f} dB. {}: {:.3f} dB. {}: {:.3f} dB. Score: {:.3f} Best{:.3f} at Epoch: {} best pnsr= {:.3f}'.format('G-PSNR', glare_psnr, 'S-PSNR', streak_psnr, 'ALL-PSNR', global_psnr, mean_psnr, best_score,best_epoch,best_psnr))

def main():

    start_epoch = 1
    if opt.resume:
        state = torch.load(models_dir + '/latest.pth')
        model.load_state_dict(state["model"])
        # optimizer.load_state_dict(state['optimizer'])
        # scheduler.load_state_dict(state['scheduler'])

        start_epoch = state['epoch'] + 1
        print('Resume model from epoch %d' % (start_epoch))

       

        # start_epoch = state['epoch'] + 1
        # print('Resume d from epoch %d' % (start_epoch))



    for epoch in range(start_epoch, opt.n_epochs + 1):
        train(epoch)

        if (epoch) % opt.val_gap == 0:
            val(epoch)
            generate(model, epoch, models_dir,test_images_dir)
            evaluate(test_images_dir, epoch)

    writer.close()

if __name__ == '__main__':
    main()


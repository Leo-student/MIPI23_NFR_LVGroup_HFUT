import time
from tqdm import tqdm
from skimage import io
from statistics import mean
from torchvision.utils import save_image
from skimage.metrics import mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils import *
import synthesis
from options import TrainOptions
from model.condition_model import NAFNet
from model.encoder_decoder import EncoderDecoder

from losses import LossCont, LossFreqReco, LossGan, LossCycleGan, LossPerceptual
from datasets import Flare_Image_Loader, SingleImgDataset

from log import Log
# log = Log(__name__).getlog()


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.models_dir, self.log_dir, self.train_images_dir, self.val_images_dir, self.test_images_dir = prepare_dir(self.opt.results_dir, self.opt.experiment, delete=(not self.opt.resume))
        self.best_score = 0
        self.best_epoch = 0 
        self.best_psnr = 0
        
        self.start_time = None
        self.end_time = None
        self.log = Log(__name__).writelog(self.opt)
        #[1]parameters preparing
        
        if opt.debug :
            self.train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/2try', crop=opt.crop)
        else :
            self.train_dataset = Flare_Image_Loader(data_source=opt.data_source + '/train', crop=opt.crop)
            
        self.fm_detection = EncoderDecoder().cuda()
        # print_para_num(self.fm_detection)
        
        
        
        # self.model = InpaintGenerator(opt).cuda()
        self.model = NAFNet().cuda()
        print_para_num(self.model)
        
        
        num_gpus = torch.cuda.device_count()
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '') 
        device_ids = [int(i) for i in range(len(cuda_visible_devices.split(','))) ]
        if opt.data_parallel:
            log.info(f'Number of available GPUs is {num_gpus}  They are"{cuda_visible_devices}" at {device_ids}.')
            self.model = nn.DataParallel(self.model, device_ids=device_ids).cuda()
            self.fm_detection = nn.DataParallel(self.fm_detection, device_ids=device_ids).cuda()
            
        else: 
            device = torch.device('cuda:0')
            self.model = self.model.to(device)  
            self.fm_detection = self.fm_detection.to(device)  
        
        
        self.criterion_cont = LossCont()
        self.criterion_fft = LossFreqReco()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [50,100], 0.5)  
        
        if self.opt.tensorboard: 
            self.writer = SummaryWriter(log_dir=self.log_dir)
        self.load()  
    def load(self):
        #load dataset 
        self.train_dataset.load_scattering_flare()
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.opt.train_bs, shuffle=True, num_workers=self.opt.num_workers, pin_memory=True, prefetch_factor = 4)
        log.info('successfully loading training pairs. =====> qty:{} bs:{}'.format(len(self.train_dataset),self.opt.train_bs))
        
        
        self.val_dataset = SingleImgDataset(data_source=self.opt.data_source + '/val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.opt.val_bs, shuffle=False, num_workers=self.opt.num_workers, pin_memory=True)    
        log.info('successfully loading validating pairs. =====> qty:{} bs:{}'.format(len(self.val_dataset),self.opt.val_bs))

        #load resume 
        if self.opt.resume:
            state = torch.load(self.models_dir + '/best.pth')
            
            # new_state_dict = {k.replace('module.', ''): v for k, v in state['model'].items()}
            # new_state_dict = {'module.' + k: v for k, v in state['model'].items()}
            self.model.load_state_dict(state["model"])
            # self.model.load_state_dict(new_state_dict)
            
            
            self.optimizer.load_state_dict(state['optimizer'])
            # self.scheduler.load_state_dict(state['scheduler'])

            start_epoch = state['epoch'] + 1
            log.info('Resume model from epoch %d' % (start_epoch))
        
        #load pretrained 
        if self.opt.pretrained is not None :
            # if not self.opt.data_parallel:
                ckp = torch.load(self.opt.pretrained)
                new_state_dict = {}
                for key, value in ckp['g'].items():
                    if key.startswith('module.'):  # 如果键以 "module." 开头，则去除该前缀
                        new_key = key[len('module.'):]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                
                self.fm_detection.load_state_dict(new_state_dict)
            # else :
            #     ckp = torch.load(self.opt.pretrained)
            #     self.fm_detection.load_state_dict(ckp['g'])
            
        
    def save(self):
        pass
        
        
    def train(self,epoch):
        self.start_time = time.time()
        self.model.train()
        max_iter = len(self.train_dataloader)
        # start_epoch = state['epoch'] + 1
        # print('Resume d from epoch %d' % (start_epoch))

        psnr_meter = AverageMeter()
        iter_cont_meter = AverageMeter()
        iter_fft_meter = AverageMeter()
        iter_re_meter = AverageMeter()
        iter_timer = Timer()
        # Calculate output of image discriminator (PatchGAN)

        with tqdm(total= len(self.train_dataloader), colour = "MAGENTA", leave=False, ncols=120 ) as pbar:
            for i, (gts, flares, imgs, _) in enumerate(self.train_dataloader):
                pbar.set_description('Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] '.format(epoch, self.opt.n_epochs, i + 1, max_iter))
            # for i, (gts, flares, imgs, _) in enumerate(train_dataloader):
                gts, flares, imgs = gts.cuda(), flares.cuda(), imgs.cuda()
                cur_batch = imgs.shape[0]
                light_source = synthesis.get_highlight_mask(flares)
                mask_gt = synthesis.flare_to_mask(flares)
                mask_lf =  -  light_source + mask_gt
                mask = self.fm_detection(imgs)
                # images_masked = (imgs * (1 - mask).float()) + mask * 
                # ------------------
                #  Train Generators
                # ------------------
                self.optimizer.zero_grad()
                
                preds_flare, preds = self.model(imgs )
                # preds_flare, preds = self.model(imgs, mask )

                loss_cont =   self.opt.lambda_flare * self.criterion_cont(preds_flare, flares).cuda() + self.criterion_cont(preds, gts).cuda()
            
                loss_fft =   self.opt.lambda_flare * self.criterion_fft(preds_flare, flares).cuda() + self.criterion_cont(preds, gts).cuda()
                
                
                masked_lf_scene = (1 - mask_lf) * imgs + mask_lf * preds
                masked_lf_flare = (1 - mask_lf) * imgs + mask_lf * preds_flare
                
                
                loss_region = self.criterion_cont(masked_lf_scene, gts).cuda()
                loss_flare = self.criterion_fft(masked_lf_flare, gts).cuda()
                
                lambda_region =   (512 * 512 * cur_batch)  / (torch.sum(mask_lf).cpu().numpy() + 1e-9) 
                # print(lambda_region, torch.sum(mask_gt), torch.sum(mask_lf), torch.sum(light_source) );print(" ")
                
                
                
                # loss =  loss_cont +  lambda_region * loss_region 
                loss =  loss_cont + self.opt.lambda_fft * loss_fft + lambda_region * loss_region 
                
                loss.backward()
                self.optimizer.step()


                iter_cont_meter.update(loss_cont.item()*cur_batch, cur_batch)
                iter_fft_meter.update(loss_fft.item()*cur_batch, cur_batch)


                # if lambda_region >= 1.0:
                    # save_image(torch.cat((imgs,preds.detach(),preds_flare.detach(),flares,gts),0), train_images_dir + '/epoch_{:0>4}_iter_{:0>4}.png'.format(epoch, i+1), nrow=opt.train_bs, normalize=True, scale_each=True)

                if self.opt.tensorboard and ((i+1) % self.opt.print_gap == 0):
                    self.writer.add_scalar('Loss_cont', iter_cont_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
                    self.writer.add_scalar('Loss_fft', iter_fft_meter.average(auto_reset=True), i+1 + (epoch - 1) * max_iter)
                
                pbar.update(1)
        if self.opt.tensorboard: 
            self.writer.add_scalar('lr', self.scheduler.get_last_lr()[0], epoch)
        
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'scheduler': self.scheduler.state_dict(), 'epoch': epoch}, self.models_dir + '/latest.pth')
        self.scheduler.step()
        
        
        
    def val(self,epoch):
        self.model.eval()
        self.fm_detection.eval()
        # print(''); 
        log.info('Validating...')

        timer = Timer()

        for i, (img, path) in enumerate(self.val_dataloader):
            img = img.cuda()

            with torch.no_grad():
                mask = self.fm_detection(img)
                pred_flare, pred = self.model(img)
                # pred_flare, pred = self.model(img, mask)
                pred_clip = torch.clamp(pred, 0, 1)
                pred_flare_clip = torch.clamp(pred_flare, 0, 1)

            if i < 5:
                # save_image(pred_clip, val_images_dir + '/epoch_{:0>4}_'.format(epoch) + os.path.basename(path[0]))
                save_image(pred_clip, self.val_images_dir + '/epoch_{:0>4}_img_'.format(epoch) + os.path.basename(path[0]), nrow=self.opt.val_bs//2, normalize=True, scale_each=True)
                save_image(pred_flare_clip, self.val_images_dir + '/epoch_{:0>4}_flare_'.format(epoch) + os.path.basename(path[0]), nrow=self.opt.val_bs//2, normalize=True, scale_each=True)
            else:
                break

        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),  'epoch': epoch}, self.models_dir + '/epoch_{:0>4}.pth'.format(epoch))

        log.info('Epoch[{:0>4}/{:0>4}] Time: {:.4f}'.format(epoch, self.opt.n_epochs, timer.timeit()))#; print('')
        
        
    def generate(self):
        infer_dataset = SingleImgDataset(data_source=self.opt.data_source + '/val')
        infer_dataloader = DataLoader(infer_dataset, batch_size=1, shuffle=False, num_workers=12, pin_memory=True)
        # print('successfully loading inferring pairs. =====> qty:{}'.format(len(infer_dataset)))
        state = torch.load(self.models_dir  + '/latest.pth')
        # model = model.module ###
        self.model.load_state_dict(state['model'])
        # model.load_state_dict(torch.load(models_dir  + '/epoch_{:0>4}.pth'.format(epoch)))


        # print('successfully loading pretrained model.')
        self.model.eval()

        time_meter = AverageMeter()
        for i, (img, path) in tqdm(enumerate(infer_dataloader, 1), colour = "YELLOW", leave=False, total=len(infer_dataloader), ncols=70):
        # for i, (img, path) in enumerate(infer_dataloader):
            img = img.cuda()

            with torch.no_grad():
                start_time = time.time()
                mask = self.fm_detection(img)
                _, pred = self.model(img)
                # _, pred = self.model(img, mask)
                pred_blend = synthesis.blend_light_source(img.cuda(), pred.cuda())
                times = time.time() - start_time

            pred_clip = torch.clamp(pred_blend, 0, 1)

            time_meter.update(times, 1)

            # print('Iteration: {:0>3}/{} Processing image...Path {} Time {:.3f}'.format((i+1) ,len(infer_dataset),path, times))


            if self.opt.save_image:
                save_image(pred_clip, self.test_images_dir + '/' + os.path.basename(path[0]))

        # print('Avg time: {:.3f}'.format(time_meter.average()))
            
    def evaluate(self, epoch):
        input_folder = os.path.join(self.test_images_dir)
        gt_folder = os.path.join(self.opt.data_source, 'val/gt')
        mask_folder = os.path.join(self.opt.data_source, 'val/mask')

        output_filename = os.path.join(self.test_images_dir, 'scores.txt')

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
       
        if self.best_score < mean_psnr :
            self.best_score = mean_psnr
            self.best_epoch = epoch
            self.best_psnr = global_psnr
            update_best = 1
        else :
            self.best_epoch = self.best_epoch
            self.best_psnr = self.best_psnr
            update_best = 0

        with open(output_filename, 'w') as f:
            f.write('{}: {:.3f}\n'.format('G-PSNR', glare_psnr))
            f.write('{}: {:.3f}\n'.format('S-PSNR', streak_psnr))
            f.write('{}: {:.3f}\n'.format('ALL-PSNR', global_psnr))
            f.write('{}: {:.3f}\n'.format('Score', mean_psnr))
            f.write('{}: {:.3f} {}\n'.format('best', self.best_score, self.best_epoch, self.best_psnr))
            # f.write('DEVICE: CPU\n')
        if self.opt.tensorboard: 
            self.writer.add_scalar('G-PSNR', glare_psnr, epoch)   
            self.writer.add_scalar('S-PSNR', streak_psnr, epoch)   
            self.writer.add_scalar('ALL-PSNR',global_psnr, epoch)   
            self.writer.add_scalar('mean_psnr',mean_psnr, epoch) 
            self.writer.add_scalar('best_psnr',self.best_psnr, epoch) 
        
        self.log.info('{}: {:.3f} dB. {}: {:.3f} dB. {}: {:.3f} dB. Score: {:.3f} Best{:.3f} at Epoch: {} best pnsr= {:.3f}'.format('G-PSNR', glare_psnr, 'S-PSNR', streak_psnr, 'ALL-PSNR', global_psnr, mean_psnr, self.best_score,self.best_epoch,self.best_psnr))
        
        if update_best:
            os.rename(self.models_dir + "/latest.pth", self.models_dir + "/best.pth")
            update_best = 0
        
   
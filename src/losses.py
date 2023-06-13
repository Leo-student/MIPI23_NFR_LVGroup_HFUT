import math
import torch
import torch.fft
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable


cuda = True if torch.cuda.is_available() else False


class LossCont(nn.Module):
    def __init__(self):
        super(LossCont, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        return self.criterion(imgs, gts)

# SmoothL1Loss
class LossSmoothL1(nn.Module):
    def __init__(self):
        super(LossSmoothL1, self).__init__()
        self.criterion = nn.SmoothL1Loss()
        
    def forward(self, imgs, gts):
        return self.criterion(imgs, gts)
# CharbonnierLoss
class LossCharbonnier(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(LossCharbonnier, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        error = prediction - target
        loss = torch.sqrt(error * error + self.epsilon * self.epsilon)
        loss = torch.mean(loss)
        return loss

class LossFreqReco(nn.Module):
    def __init__(self):
        super(LossFreqReco, self).__init__()
        self.criterion = nn.L1Loss()
        
    def forward(self, imgs, gts):
        imgs = torch.fft.rfftn(imgs, dim=(2,3))
        _real = imgs.real
        _imag = imgs.imag
        imgs = torch.cat([_real, _imag], dim=1)
        gts = torch.fft.rfftn(gts, dim=(2,3))
        _real = gts.real
        _imag = gts.imag
        gts = torch.cat([_real, _imag], dim=1)
        return self.criterion(imgs, gts)
    
class LossGan(nn.Module):
    def __init__(self):
        super(LossGan, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, imgs, gts):
        return self.criterion(imgs, gts)
class LossCycleGan(nn.Module):
    def __init__(self):
        super(LossCycleGan, self).__init__()
        self.criterion = nn.MSELoss()
        
    def forward(self, imgs, gts):
        return self.criterion(imgs, gts)
    
    


class LossPerceptual(nn.Module):
    def __init__(self,):
        super(LossPerceptual, self).__init__()
        # self.device = device
        self.vgg = models.vgg19(pretrained= True).features
        # self.vgg = models.vgg19(weights="/home/workspace/flare7K/naf/src/vgg19-dcbb9e9d.pth").features
        # self.vgg = models.vgg19(pretrained=True).features
        # self.vgg.cuda()
        self.vgg.to("cuda")
        self.criterion = nn.MSELoss()

        # 去掉 VGG 的最后一层
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        # 冻结 VGG 的所有参数
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, fake_img, real_img):
        # 获取 VGG 的中间层特征
        fake_features = self.vgg(fake_img)
        real_features = self.vgg(real_img)

        # 计算 MSE 损失
        loss = self.criterion(fake_features, real_features)

        return loss
    

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, prediction, target):
        error = prediction - target
        loss = torch.sqrt(error * error + self.epsilon * self.epsilon)
        loss = torch.mean(loss)
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):     
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]                                       
    
class LossVGGInfoNCE(nn.Module):
    def __init__(self):
        super(LossVGGInfoNCE, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        
    def forward(self, out, gt, input):
        
        loss = self.infoNCE(out, gt, input)
        return loss
    
    def infer(self, x):
        return self.vgg(x)
    
    def infoNCE(self, pred, gt, flared):
        
        with torch.no_grad():
            pred, gt, flared = self.infer(pred), self.infer(gt), self.infer(flared)
            loss = 0
            
            ap_dist, an_dist = 0, 0
            for i in range(len(pred)):
                ap_dist = self.l1(pred[i], gt[i].detach())
            
                an_dist = self.l1(pred[i], flared[i].detach())
                contrastive = ap_dist / (an_dist + 1e-7)
            

            loss += self.weights[i] * contrastive
        return loss
            

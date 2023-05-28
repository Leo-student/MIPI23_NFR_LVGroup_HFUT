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
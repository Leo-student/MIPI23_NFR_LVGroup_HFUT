import sys
import os

# 获取当前脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父级目录的绝对路径
parent_dir = os.path.dirname(script_dir)
# 将父级目录添加到模块搜索路径中
sys.path.append(parent_dir)


import torch
import torch.nn as nn
import torch.nn.functional as F

from options import TrainOptions
import torchinfo
opt = TrainOptions().parse(show =False)

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


class UNetD(nn.Module):

    def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2, subspace_dim = 16):
    # def __init__(self, in_chn, wf=32, depth=5, relu_slope=0.2, subspace_dim = 16):
        super(UNetD, self).__init__()

        self.depth = depth
        self.down_path = nn.ModuleList()
        # self.down_path = []

        prev_channels = self.get_input_chn(in_chn)

        for i in range(depth):
            downsample = True if (i+1) < depth else False
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope, opt))
            prev_channels = (2**i) * wf
            # print(f'prev_channels {prev_channels}')

        # self.ema = EMAU(prev_channels, prev_channels//8)
        # self.up_path = []
        self.middle = nn.Sequential(*[AOTBlock(prev_channels, opt.rates) for _ in range(opt.num_aot)])
        
        self.up_path = nn.ModuleList()

        subnet_repeat_num = 1
        for i in reversed(range(depth - 1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, subnet_repeat_num, subspace_dim))
            prev_channels = (2**i)*wf
            subnet_repeat_num += 1

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        #self._initialize()

    def forward(self, x1):

        blocks = []
        for i, down in enumerate(self.down_path):
            # print(x1.shape)
            if (i+1) < self.depth:
                x1, x1_up = down(x1)
                blocks.append(x1_up)
            else:
                x1 = down(x1)
        # # print(x1.shape)
        # x1 = self.ema(x1)
        x1 = self.middle(x1)
        for i, up in enumerate(self.up_path):
            # # print(x1.shape, blocks[-i-1].shape)
            x1 = up(x1, blocks[-i-1])

        pred = self.last(x1)
        return pred

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print("weight")
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # print("bias")
                    nn.init.zeros_(m.bias)


class UNetConvBlock(nn.Module):

    def __init__(self, in_size, out_size, downsample, relu_slope, opt):
        super(UNetConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            # nn.GELU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))
        # print(f'insize {in_size} out_size = {out_size}')
        # self.block = nn.Sequential(AOTBlock(in_size, opt.rates) )   
        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)
        # self.shortcut = nn.Sequential(*[AOTBlock(in_size,out_size , opt.rates) for _ in range(opt.num_res)])
    def forward(self, x):

        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16 ):
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope, opt)
        self.num_subspace = subspace_dim
        # print(self.num_subspace, subnet_repeat_num)
        
        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)

    def forward(self, x, bridge):
        up = self.up(x)
        bridge = self.skip_m(bridge)
        out = torch.cat([up, bridge], 1)

        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_*w_)
            V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdim=True))
            V = V_t.transpose(1, 2)
            mat = torch.matmul(V_t, V)
            mat_inv = torch.inverse(mat)
            project_mat = torch.matmul(mat_inv, V_t)
            bridge_ = bridge.reshape(b_, c_, h_*w_)
            project_feature = torch.matmul(project_mat, bridge_.transpose(1, 2))
            bridge = torch.matmul(V, project_feature).transpose(1, 2).reshape(b_, c_, h_, w_)
            out = torch.cat([up, bridge], 1)
        
        out = self.conv_block(out)
        return out


class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()

        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2, opt))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)

        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()

        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2, opt))

        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2, opt))

        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2, opt))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)

        for m in self.blocks:
            x = m(x)
        return x + sc

class AOTBlock(nn.Module):
    def __init__(self, dim_in, rates):
        super(AOTBlock, self).__init__()
        self.rates = rates
        
        
        for i, rate in enumerate(rates):
            
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)), 
                nn.Sequential(
                    nn.ReflectionPad2d(int(rate)),
                    nn.Conv2d(dim_in, dim_in//len(self.rates), 3, padding=0, dilation=int(rate)),
                    nn.ReLU(True)))
        self.fuse = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in,dim_in ,3, padding=0, dilation=1))
        self.gate = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim_in, dim_in,3, padding=0, dilation=1))

    def forward(self, x):
        
        out = []
        for i  in range(len(self.rates)):
            block_name = f'block{str(i).zfill(2)}'
            
            
            block = self.__getattr__(block_name)
            # print(f"Block {i} output shape: {x.shape}")
            input_channels = x.shape[1]
            input_channels_block = block[1].in_channels
            output_channels = block[1].out_channels
            # print(f"Block {i} - Input Channels: {input_channels}, in: {input_channels_block} Output Channels: {output_channels}")
            
            block_output = block(x)
            out.append(block_output)
            # print(f"Block {i} output shape: {block_output.shape}")

        # out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        # print(f"Concatenated output shape: {out[0].shape, out[1].shape}")
        out = torch.cat(out, 1)
        # print(f"Concatenated output shape: {out.shape}")
        
        out = self.fuse(out)
        # print(f"Fused output shape: {out.shape}")
        
        mask = my_layer_norm(self.gate(x))
        # print(f"Mask shape: {mask.shape}")
        
        
        mask = torch.sigmoid(mask)
        # print(f"Sigmoid Mask shape: {mask.shape}")
        
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat

if __name__ == "__main__":
    import numpy as np
    a = UNetD(3)

    ## print(a)
    input_size = (4, 3, 512, 512)
    input_data = torch.randn(input_size)
    torchinfo.summary(a, input_size)
    
    # im = mge.tensor(np.random.randn(1, 3, 128, 128).astype(np.float32))
    # # print(a(im))


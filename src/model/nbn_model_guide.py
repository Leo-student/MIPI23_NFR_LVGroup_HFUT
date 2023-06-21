import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

class ConvGuidedFilter(nn.Module):
    def __init__(self, in_size, out_size,radius=1, norm=nn.BatchNorm2d):
        super(ConvGuidedFilter, self).__init__()

        self.box_filter = nn.Conv2d(in_size, out_size, kernel_size=3, padding=radius, dilation=radius, bias=False, groups=4)
        self.conv_a = nn.Sequential(nn.Conv2d(in_size , 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, 32, kernel_size=1, bias=False),
                                    norm(32),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(32, out_size, kernel_size=1, bias=False))
        self.box_filter.weight.data[...] = 1.0

    def forward(self, x_lr, y_lr, x_hr):
        _, c_lrx, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()

        N = self.box_filter(x_lr.data.new().resize_((1, c_lrx, h_lrx, w_lrx)).fill_(1.0))
        # # print(N)
        ## mean_x
        mean_x = self.box_filter(x_lr)/N
        ## mean_y
        mean_y = self.box_filter(y_lr)/N
        ## cov_xy
        cov_xy = self.box_filter(x_lr * y_lr)/N - mean_x * mean_y
        ## var_x
        var_x  = self.box_filter(x_lr * x_lr)/N - mean_x * mean_x
        # print(cov_xy.shape , var_x.shape)
        ## A
        A = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        ## b
        b = mean_y - A * mean_x

        ## mean_A; mean_b # 最后变成高分辨率的系数 
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode='bilinear', align_corners=True)
        #返回值 会和 a * 输入  + b ，没有将高分辨率的图进行操作，但是网络输入的时候会把这个图也输入进来 ， 那也可以不输入进来
        return mean_A * x_hr + mean_b

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
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*wf, downsample, relu_slope))
            prev_channels = (2**i) * wf

        # self.ema = EMAU(prev_channels, prev_channels//8)
        # self.up_path = []

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
            # print(f'encodoer  {x1.shape}')
            if (i+1) < self.depth:
                x_hr = x1
                x1, x1_up = down(x1)
                blocks.append(x1_up)
                
                
            else:
                x1 = down(x1)
        # print(f'bottleneck {x1.shape}')
        # x1 = self.ema(x1)
        for i, up in enumerate(self.up_path):
            # print(f'decoder {x1.shape}, {blocks[-i-1].shape}')
            # bridge :    blocks[-i-1] : INH 
            #        :    x1           : INL
            x1 = up( x1, blocks[-i-1])

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

    def __init__(self, in_size, out_size, downsample, relu_slope):
        super(UNetConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(relu_slope))

        self.downsample = downsample
        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):

        out = self.block(x)
        sc = self.shortcut(x)
        out = out + sc

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out
        
class AdaptiveNorm(nn.Module):
    def __init__(self, n):
        super(AdaptiveNorm, self).__init__()

        self.w_0 = nn.Parameter(torch.Tensor([1.0]))
        self.w_1 = nn.Parameter(torch.Tensor([0.0]))

        self.bn  = nn.BatchNorm2d(n, momentum=0.999, eps=0.001)

    def forward(self, x):
        return self.w_0 * x + self.w_1 * self.bn(x)
# class DeepGuidedFilterConvGF(nn.Module):
#     def __init__(self, radius=1, layer=5):
#         super(DeepGuidedFilterConvGF, self).__init__()
        
#         self.gf = ConvGuidedFilter(radius, norm=AdaptiveNorm)

#     def forward(self, x_lr, x_hr):
#         return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

   
class UNetUpBlock(nn.Module):

    def __init__(self, in_size, out_size, relu_slope, subnet_repeat_num, subspace_dim=16):
        super(UNetUpBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)
        self.conv_equal = UNetConvBlock(in_size, in_size, False, relu_slope)
        self.num_subspace = subspace_dim
        # print(self.num_subspace, subnet_repeat_num)
        
        self.subnet = Subspace(in_size, self.num_subspace)
        self.skip_m = skip_blocks(out_size, out_size, subnet_repeat_num)
        
        self.gf = ConvGuidedFilter(in_size, out_size,radius = 1, norm=AdaptiveNorm)
    #              x_lr y_lr x_hr
    def forward(self, x, bridge):
        x_lr = x 
        up = self.up(x)
        y_lr = self.conv_equal(x_lr)
        
        
        
        bridge = self.skip_m(bridge)
        # print(x_lr.shape, y_lr.shape, bridge.shape)
        y_hr = self.gf(x_lr, y_lr, bridge)
        
        out = torch.cat([up, y_hr], 1)

        if self.subnet:
            b_, c_, h_, w_ = bridge.shape
            sub = self.subnet(out)
            V_t = sub.reshape(b_, self.num_subspace, h_*w_)
            V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdim=True))
            V = V_t.transpose(1, 2)
            mat = torch.matmul(V_t, V)
            try :
                mat_inv = torch.inverse(mat)
            except: 
                mat_inv = torch.pinverse(mat)
                print()
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

        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
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
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))

        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))

        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)

        for m in self.blocks:
            x = m(x)
        return x + sc


if __name__ == "__main__":
    import numpy as np
    a = UNetD(3)

    # print(a)
    input_size = (4, 3, 512, 512)
    input_data = torch.randn(input_size)
    torchinfo.summary(a, input_size)
    im = torch.tensor(np.random.randn(4, 3, 512, 512).astype(np.float32))
    # # print(a(im))


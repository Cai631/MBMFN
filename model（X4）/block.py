import torch.nn as nn
from collections import OrderedDict
import torch

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias, dilation=dilation,
                     groups=groups)

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer

def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)
                                                                                   # 定义 hwish  jiade
'''class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super(HardSwish, self).__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return x * (self.relu6(x+3)) / 6'''

def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        #layer = HardSwish(inplace)                                               #    xiu gai

        layer = nn.LeakyReLU(neg_slope, inplace)

    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer

class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output

def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class CCALayer(nn.Module):
    def __init__(self, channel):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential( nn.Conv2d(channel, channel, 1, padding=0, bias=True),nn.Sigmoid())
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y+x

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential( nn.Conv2d(channel, channel, 3, padding=(3 - 1) // 2, bias=True),nn.Sigmoid())
        self.c3 = nn.Conv2d(channel, channel, kernel_size=3, padding=(3 - 1) // 2, bias=False)
    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        x = self.c3(x)
        return x * y
class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):
        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.conv(x)
        y = self.sigmoid(y)
        out = torch.mul(x, y)
        return out
class PAConv(nn.Module):
    def __init__(self, nf, k_size=3):
        super(PAConv, self).__init__()
        self.k2 = nn.Conv2d(nf, nf, 1) # 1x1 convolution nf->nf
        self.sigmoid = nn.Sigmoid()
        self.k3 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
        self.k4 = nn.Conv2d(nf, nf, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) # 3x3 convolution
    def forward(self, x):
        y = self.k2(x)
        y = self.sigmoid(y)
        out = torch.mul(self.k3(x), y)
        out = self.k4(out)
        return out
    
class IMDModule(nn.Module):
    def __init__(self, in_channels, distillation_rate1=0.25,distillation_rate2=0.5,distillation_rate3=0.75):
        super(IMDModule, self).__init__()
        self.distilled_channels1 = int(in_channels * distillation_rate1)  # 0.25
        self.distilled_channels2 = int(in_channels * distillation_rate2)  #0.5
        self.distilled_channels3 = int(in_channels * distillation_rate3)  # 0.75
        self.distilled_channels4 = int(in_channels * 3)  # 3
        
        self.hc1 = conv_layer(in_channels, self.distilled_channels1, 1)
        self.act = activation('relu')

        self.c11 = conv_layer(self.distilled_channels1, self.distilled_channels1, 3)
        self.c12 = conv_layer(self.distilled_channels1, self.distilled_channels1, 3)
        self.ccaLayer1 = CCALayer(self.distilled_channels1)

        self.c21 = conv_layer(self.distilled_channels2, self.distilled_channels2, 3)
        self.c22 = conv_layer(self.distilled_channels2, self.distilled_channels2, 3)
        self.ccaLayer2 = CCALayer(self.distilled_channels2)

        self.c31 = conv_layer(self.distilled_channels3, self.distilled_channels3, 3)
        self.c32 = conv_layer(self.distilled_channels3, self.distilled_channels3, 3)
        self.ccaLayer3 = CCALayer(self.distilled_channels3)

        self.c1 = conv_layer(in_channels, in_channels, 3)
        self.c3 = conv_layer(in_channels, in_channels, 3)
        self.ccaLayer = CCALayer(in_channels)

        self.last1 = conv_layer(self.distilled_channels4, in_channels, 1)
        self.last3 = conv_layer(in_channels, in_channels, 3)

    def forward(self, input):
        outh = self.hc1(input)
        out1 = self.c12(self.act(self.c11(outh)))
        out12 = out1+outh
        out13 = self.ccaLayer1(out12)

        out2 = torch.cat([outh, out12], dim=1)
        out21 = self.c22(self.act(self.c21(out2)))
        out22 = out21+out2
        out23 = self.ccaLayer2(out22)

        out3 = torch.cat([outh, out22], dim=1)
        out31 = self.c32(self.act(self.c31(out3)))
        out32 = out31 + out3
        out33 = self.ccaLayer3(out32)

        out4 = torch.cat([outh, out32], dim=1)
        out41 = self.c3(self.act(self.c1(out4)))
        out42 = out41 + out4
        out43 = self.ccaLayer(out42)

        out5 = torch.cat([out13, out23,out33,out43,out22], dim=1)
        out51 = self.act(self.last3(self.last1(out5)))
        out52 = self.ccaLayer(out51)
        out53 = out52+input
        return out53
        
        
        
        
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

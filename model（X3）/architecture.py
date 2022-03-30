import torch.nn as nn
from . import block as B
import torch
import torch.nn.functional as F

class IMDN(nn.Module):
    def __init__(self, in_nc=3, nf=64, num_modules=6, out_nc=3, upscale=3):
        super(IMDN, self).__init__()
        self.upscale = upscale 
        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=3)
        self.act = B.activation('lrelu', neg_slope=0.05)
        

        # IMDBs
        self.IMDB1 = B.IMDModule(in_channels=nf)
        self.IMDB2 = B.IMDModule(in_channels=nf)
        self.IMDB3 = B.IMDModule(in_channels=nf)
        self.IMDB4 = B.IMDModule(in_channels=nf)
        self.IMDB5 = B.IMDModule(in_channels=nf)
        self.IMDB6 = B.IMDModule(in_channels=nf)
        
        #self.up1 = B.PA(nf=64)
        self.up1 = B.CCALayer(channel=64)
        self.LR_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.HR_conv = B.conv_layer(nf, 3, kernel_size=3)                      # jia de

    def forward(self, input):
        out_fea = self.act(self.fea_conv(input))                                      #  加的

        out_B1 = self.IMDB1(out_fea)
        out_B2 = self.IMDB2(out_B1)
        out_B3 = self.IMDB3(out_B2)
        out_B4 = self.IMDB4(out_B3)
        out_B5 = self.IMDB5(out_B4)
        out_B6 = self.IMDB6(out_B5)
        fea = self.LR_conv(F.interpolate(out_B6, scale_factor=2, mode='nearest'))
        fea = self.act(self.up1(fea))
        fea = (self.LR_conv(fea)) 
        fea = self.LR_conv(F.interpolate(fea, scale_factor=1.5, mode='nearest'))
        fea = self.act(self.up1(fea))
        fea = (self.LR_conv(fea))
        out = self.HR_conv(fea) + F.interpolate(input, scale_factor=self.upscale, mode='bilinear', align_corners=False) # 
        return out
        

        #out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        #out_lr = self.act(self.LR_conv(out_B))                                                     #+ out_fea0
        #output = self.upsampler(out_lr)
        #output = self.HR_conv(output)                                                  #  jia  de
        


# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       main
   Project Name:    beamform_Ynet
   Author :         Hengrong LAN
   Date:            2018/12/27
   Device:          GTX1080Ti
-------------------------------------------------
   Change Activity:
                   2018/12/10:
-------------------------------------------------
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import numpy as np




def conv3x3(in_channels, out_channels, stride=1, 
            padding=1, bias=True, groups=1):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    if mode == 'transpose':
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        # out_channels is always going to be the same
        # as in_channels
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)



class DownConv(nn.Module):

    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        before_pool = x
        if self.pooling:
            x = self.pool(x)
        return x, before_pool

class Bottom(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(Bottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels,
                               kernel_size=(20, 3), stride=(20, 1),padding=(0,1))
        self.conv3 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.01)

        return x

class ImBottom(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(ImBottom, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)

        return x

class UpConv(nn.Module):

    def __init__(self, in_channels, out_channels, 
                 merge_mode='add', up_mode='transpose'):
        super(UpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode

        self.upconv = upconv2x2(self.in_channels, self.in_channels,
            mode=self.up_mode)

        if self.merge_mode == 'add':
            self.conv1 = conv3x3(
                self.in_channels, self.out_channels)
        else:
            # num of input channels to conv2 is same,concat
            self.conv1 = conv3x3(2*self.in_channels, self.out_channels)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)


    def forward(self, from_down, from_up):
        """ Forward pass
        Arguments:

            from_down: tensor from the das encoder pathway
            from_up: upconv'd tensor from the decoder pathway
        """
        from_up = self.upconv(from_up)
        if self.merge_mode == 'add':
            x = from_up + from_down

        else:
            #concat
            x = torch.cat((from_up, from_down), 1)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        return x


# Model 1 Ynet:2 concat: raw data cat
class YNet(nn.Module):


    def __init__(self,  in_channels=3, up_mode='transpose', merge_mode='concat'):
        """
        Arguments:
            in_channels: int, number of channels in the input tensor.
                Default is 3 for RGB images.
            up_mode: string, type of upconvolution. Choices: 'transpose'
                for transpose convolution or 'upsample' for nearest neighbour
                upsampling.
        """
        super(YNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
    
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))

        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neighbour to reduce "
                             "depth channels (by half).")

        self.in_channels = in_channels
       
        self.down1 = DownConv(1,32)
        self.down2 = DownConv(32,64)
        self.down3 = DownConv(64,128)
        self.down4 = DownConv(128,256)
        self.bottom = Bottom(256,256)
        self.bbottom = ImBottom(256,256)
        self.combine = ImBottom(512,256)
        self.up1 = UpConv(256,128,merge_mode=self.merge_mode)
        self.up2 = UpConv(128,64,merge_mode=self.merge_mode)
        self.up3 = UpConv(64,32,merge_mode=self.merge_mode)
        self.up4 = UpConv(32,1,merge_mode=self.merge_mode)
        self.up5 = UpConv(1,1,merge_mode=self.merge_mode)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)


    def forward(self, x,bfimg):
         #encoder1: raw data
        x1, before_pool1 = self.down1(x) # 1280, 64,32
        x2, before_pool2 = self.down2(x1)# 640, 32,64
        x3, before_pool3 = self.down3(x2) # 320, 16,128
        x4, before_pool4 = self.down4(x3) # 160, 8,256
        x5 = self.bottom(x4)
        before_pool4_resize = F.upsample(before_pool4, (16, 16), mode='bilinear')
        before_pool3_resize = F.upsample(before_pool3, (32, 32), mode='bilinear')
        before_pool2_resize = F.upsample(before_pool2, (64, 64), mode='bilinear')
        before_pool1_resize = F.upsample(before_pool1, (128, 128), mode='bilinear')
         #encoder2: bf
        bx1,_= self.down1(bfimg)
        bx2,_= self.down2(bx1)
        bx3,_= self.down3(bx2)
        bx4,_= self.down4(bx3)
        bx5 = self.bbottom(bx4) # 8, 8, 256
        if self.merge_mode == 'add':
            out = x5 + bx5

        else:
            #concat
            out = torch.cat((x5, bx5), 1)
        out = self.combine(out)
        out = self.up1(before_pool4_resize, out)# 16, 16,128
        out = self.up2(before_pool3_resize, out)# 32, 32,64
        out = self.up3(before_pool2_resize, out)# 64, 64,32
        out = self.up4(before_pool1_resize, out)# 128, 128,1
        return out

if __name__ == "__main__":
    """
    testing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device =  torch.device('cuda:1')


    x = Variable(torch.FloatTensor(np.random.random((1, 1, 2560, 128))),requires_grad = True).to(device)
    img = Variable(torch.FloatTensor(np.random.random((1, 1, 128, 128))), requires_grad=True).to(device)
    model = YNet(in_channels=1,merge_mode='concat').to(device)
    out = model(x,img)
    out = F.upsample(out, (128, 128), mode='bilinear')
    loss = torch.mean(out)

    loss.backward()

    print(loss)

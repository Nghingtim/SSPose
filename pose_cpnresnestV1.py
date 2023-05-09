# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import logging

import torch
import torch.nn as nn

import math

import sys
sys.path.insert(0, '/home/cair2016/Projects/deep-high-resolution-net.pytorch/lib/models/ResNeSt/resnest/torch')

from resnest import resnest50
 
from splat import SplAtConv2d

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)

import torch.nn.functional as F



import re

import torch.utils.checkpoint as cp
from collections import OrderedDict
#from .utils import load_state_dict_from_url

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
from torch import Tensor
from typing import Any, List, Tuple
 


def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, out_planes=0, stride=1):
        super(Bottleneck, self).__init__()
        out_planes = self.expansion*planes if out_planes == 0 else out_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class PoseHGResNeSt(nn.Module):
    def __init__(self, cfg, pretrained=True, num_keypoints=17):
        super(PoseHGResNeSt, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.inplanes = 2048   
        self.num_keypoints =num_keypoints          
        #model = densenet121(pretrained=pretrained)
        
        model = resnest50(pretrained=True)
        
        #self.model = nn.Sequential(*list(model.children())[:-2])
       	self.layer1 = nn.Sequential(*list(model.children())[:-5])
        self.layer2 = nn.Sequential(*list(model.children())[-5])
        self.layer3 = nn.Sequential(*list(model.children())[-4])
        self.layer4 = nn.Sequential(*list(model.children())[-3])
        
        self.lateral1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.lateral2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.lateral3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.lateral4 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        self.smooth1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.global1 = self._make_global(scale_factor=8)
        self.global2 = self._make_global(scale_factor=4)
        self.global3 = self._make_global(scale_factor=2)
        self.global4 = self._make_global(scale_factor=1)

        self.refine1 = self._make_refine(num_blocks=3, scale_factor=8)  #input 256 8 8
        self.refine2 = self._make_refine(num_blocks=2, scale_factor=4)
        self.refine3 = self._make_refine(num_blocks=1, scale_factor=2)
        self.refine4 = nn.Sequential(
            Bottleneck(4*256, 128, 256),
            nn.Conv2d(256, num_keypoints, kernel_size=3, stride=1, padding=1),
        )

        # self.splitAttention = self._make_splitAttention(8)
        self.splitAttention = SplAtConv2d(
                in_channels=512, channels=256, kernel_size=3,
                stride=1, padding=1,
                dilation=1, groups=1, bias=False,
                radix=2, rectify=False,
                rectify_avg=False,
                norm_layer=nn.BatchNorm2d,
                dropblock_prob=0.0)

        self.deconv_with_bias = extra.DECONV_WITH_BIAS
              


        # used for deconv layers
        deconv_layers = self._make_deconv_layer(
            extra.NUM_DECONV_LAYERS,
            extra.NUM_DECONV_FILTERS,
            extra.NUM_DECONV_KERNELS,
        )
        
        self.deconv1 =  nn.Sequential(*list(deconv_layers.children())[:3])
        self.deconv2 =  nn.Sequential(*list(deconv_layers.children())[3:6])
        self.deconv3 =  nn.Sequential(*list(deconv_layers.children())[6:9])

      
        self.downConv1 = self._make_downConv(256, 256, 3, 1)
        self.downConv2 = self._make_downConv(512, 256, 3,1)
        self.downConv3 = self._make_downConv(1024, 256, 3,1)

        ####??SENet??
        #
        #self.selayer = SELayer(256)
        
        
        self.final_layer = nn.Conv2d(
            in_channels=extra.NUM_DECONV_FILTERS[-1],
            out_channels=cfg.MODEL.NUM_JOINTS,
            kernel_size=extra.FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0
        )
        

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                c1, c2, h, w = m.weight.data.size()
                weight = get_upsample_filter(h)
                m.weight.data = weight.view(1, 1, h, w).repeat(c1, c2, 1, 1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_splitAttention(self, splitNum):
        layers = []
        for i in range(depth):
            if i > 0:
                inplanes = outplanes
            layers.append(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1))
            layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers) 

    def _make_downConv(self, inplanes, outplanes, kernel_size, depth):
        layers = []
        for i in range(depth):
            if i > 0:
                inplanes = outplanes
            layers.append(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=1))
            layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers) 
        
    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

 
    def _make_global(self, scale_factor):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            # nn.Conv2d(256, self.num_keypoints, kernel_size=3, stride=1, padding=1),  #去掉17
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),   ##以后可以考虑加入反卷积
        )

    def _make_refine(self, num_blocks, scale_factor):
        layers = []
        for i in range(num_blocks):
            layers.append(Bottleneck(256,128,256))
        layers.append(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False))
        return nn.Sequential(*layers)

    def _upsample_smooth_add(self, x, smooth, y):
        up = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=False)
        return smooth(up) + F.relu(y)

    def forward(self, x):
        # Top-down
        # c1 = F.relu(self.bn1(self.conv1(x)))
        # c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(x)  #256 64 64
        c3 = self.layer2(c2) #512 32 32
        c4 = self.layer3(c3) # 1024 16 16
        c5 = self.layer4(c4)  #2048 8 8
        # Bottom-up
        p5 = self.lateral1(c5)  #256 8 8
        p4 = self._upsample_smooth_add(p5, self.smooth1, self.lateral2(c4)) #256 16 16
        p3 = self._upsample_smooth_add(p4, self.smooth2, self.lateral3(c3)) #256 32 32
        p2 = self._upsample_smooth_add(p3, self.smooth3, self.lateral4(c2)) #256 64 64
        # GlobalNet
        g5 = self.global1(p5) #17 64 64
        g4 = self.global2(p4)  #17 64 64
        g3 = self.global3(p3)  #17 64 64
        g2 = self.global4(p2)  #17 64 64 ---》256 64 64
        g = g5+g4+g3+g2
        
        # RefineNet
        r5 = self.refine1(p5)
        r4 = self.refine2(p4)
        r3 = self.refine3(p3)
        r2 = p2  #都是256 64 64
        r = r5+r4+r3+r2

        #r = torch.cat([g5, g4, g3, g2, r5,r4,r3,r2], 1)  #1024 64 64 --->256*8 64 64
        r = torch.cat([g, r], 1)  #512 64 64 --->256*2 64 64
        
        sa = self.splitAttention(r)  #256 64 64

        # r = torch.cat([r5,r4,r3,r2], 1)  #1024 64 64 
        # r = self.refine4(r)  #17 64 64
        # return g5, g4, g3, g2, r

        x = self.final_layer(sa)  #16 64 64
        
        #x = self.model(x)
        #print(x.shape)
        #x = self.deconv_layers(x)
        #print(x.shape)
        
        return x  

    def forwardV0(self, x): 
    
        x1 = self.Layer1(x)  #256 64 64
        #print(x1.shape)
	
        x2 = self.Layer2(x1)  #512 32 32
        #print(x2.shape)

        x3 = self.Layer3(x2)
        #print(x3.shape)

        x4 = self.Layer4(x3)
        #print(x4.shape)

        y1 = self.deconv1(x4)  #1024 16 16
        #print(y1.shape)
	      
        y2 = y1 + self.downConv3(x3)
        
        y3 = self.deconv2(y2)
        #print(y3.shape)
        
        y4 = y3 + self.downConv2(x2)

        y5 = self.deconv3(y4)
        #print(y5.shape)
        
        x = y5 + self.downConv1(x1) #256 64 64
        
        ####??SELayer
        #x = self.selayer(x)  #256 64 64

        x = self.final_layer(x)  #16 64 64
        
        #x = self.model(x)
        #print(x.shape)
        #x = self.deconv_layers(x)
        #print(x.shape)
        
        #x = self.final_layer(x)
        return x    
                    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('V3---HGResNeSt50=> init deconv weights from normal distribution')
            for name, m in self.deconv1.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for name, m in self.deconv2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for name, m in self.deconv3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            logger.info('=> init down-conv weights from normal distribution')
            for name, m in self.global1.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for name, m in self.global2.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for name, m in self.global3.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for name, m in self.global4.named_modules():
                if isinstance(m, nn.Conv2d):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                elif isinstance(m, nn.BatchNorm2d):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.info('PoseDenseNet=> init weights from normal distribution')
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    nn.init.normal_(m.weight, std=0.001)
                    # nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False
    )


def test():
    net = densenet_pose()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)
    
def get_pose_net(cfg, is_train, **kwargs):

    #return DensePose(cfg)  ### PreDensePose on MPII for COCO
    
    model =  PoseHGResNeSt(cfg)
    
    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
    
    #return DensePose(cfg)
    #num_layers = cfg.MODEL.EXTRA.NUM_LAYERS

    #block_class, layers = resnet_spec[num_layers]
    

    model = PoseDenseNet(Bottleneck, [6,12,32,32], cfg, growth_rate=32)  #PoseDenseNet(block_class, layers, cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
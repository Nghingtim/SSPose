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

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


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

class PoseHGResNeSt(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super(PoseHGResNeSt, self).__init__()
        extra = cfg.MODEL.EXTRA
        self.inplanes = 2048             
        #model = densenet121(pretrained=pretrained)
        
        model = resnest50(pretrained=True)
        
        #self.model = nn.Sequential(*list(model.children())[:-2])
       	self.Layer1 = nn.Sequential(*list(model.children())[:-5])
        self.Layer2 = nn.Sequential(*list(model.children())[-5])
        self.Layer3 = nn.Sequential(*list(model.children())[-4])
        self.Layer4 = nn.Sequential(*list(model.children())[-3])


        self.downConv1 = self._make_downConv(256, 256, 3)
        self.downConv2 = self._make_downConv(512, 256, 2)
        self.downConv3 = self._make_downConv(1024, 256, 1)
        
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

    def _make_downConv(self, inplanes, outplanes, depth):
        layers = []
        for i in range(depth):
            if i > 0:
                inplanes = outplanes
            layers.append(
                nn.Conv2d(
                    in_channels=inplanes,
                    out_channels=outplanes,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            layers.append(nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers) 
                       

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
        
    def forward(self, x): 
    
        x1 = self.Layer1(x)
        #print(x1.shape)
	
        x2 = self.Layer2(x1)
        #print(x2.shape)

        x3 = self.Layer3(x2)
        #print(x3.shape)

        x4 = self.Layer4(x3)
        #print(x4.shape)

        y1 = self.deconv1(x4)
        #print(y1.shape)
	      
        y2 = y1 + self.downConv3(x3)  #16*16
        
        y3 = self.deconv2(y2)  #32*32
        #print(y3.shape)
        
        y4 = y3 + self.downConv2(x2)

        y5 = self.deconv3(y4) #64*64
        #print(y5.shape)
        
        y6 = y5 + self.downConv1(x1)

        x = self.final_layer(y6)
        
        #x = self.model(x)
        #print(x.shape)
        #x = self.deconv_layers(x)
        #print(x.shape)
        
        #x = self.final_layer(x)
        return x    
                    
    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('PoseHGResNeSt50=> init deconv weights from normal distribution')
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
# Modified from
# https://github.com/NVlabs/SegFormer/blob/master/mmseg/models/decode_heads/segformer_head.py
#
# This work is licensed under the NVIDIA Source Code License.
#
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)
#
#  1. Definitions
#  "Licensor" means any person or entity that distributes its Work.
#  "Software" means the original work of authorship made available under
# this License.
#  "Work" means the Software and any additions to or derivative works of
# the Software that are made available under this License.
#  The terms "reproduce," "reproduction," "derivative works," and
# "distribution" have the meaning as provided under U.S. copyright law;
# provided, however, that for the purposes of this License, derivative
# works shall not include works that remain separable from, or merely
# link (or bind by name) to the interfaces of, the Work.
#  Works, including the Software, are "made available" under this License
# by including in or with the Work either (a) a copyright notice
# referencing the applicability of this License to the Work, or (b) a
# copy of this License.
#  2. License Grants
#      2.1 Copyright Grant. Subject to the terms and conditions of this
#     License, each Licensor grants to you a perpetual, worldwide,
#     non-exclusive, royalty-free, copyright license to reproduce,
#     prepare derivative works of, publicly display, publicly perform,
#     sublicense and distribute its Work and any resulting derivative
#     works in any form.
#  3. Limitations
#      3.1 Redistribution. You may reproduce or distribute the Work only
#     if (a) you do so under this License, (b) you include a complete
#     copy of this License with your distribution, and (c) you retain
#     without modification any copyright, patent, trademark, or
#     attribution notices that are present in the Work.
#      3.2 Derivative Works. You may specify that additional or different
#     terms apply to the use, reproduction, and distribution of your
#     derivative works of the Work ("Your Terms") only if (a) Your Terms
#     provide that the use limitation in Section 3.3 applies to your
#     derivative works, and (b) you identify the specific derivative
#     works that are subject to Your Terms. Notwithstanding Your Terms,
#     this License (including the redistribution requirements in Section
#     3.1) will continue to apply to the Work itself.
#      3.3 Use Limitation. The Work and any derivative works thereof only
#     may be used or intended for use non-commercially. Notwithstanding
#     the foregoing, NVIDIA and its affiliates may use the Work and any
#     derivative works commercially. As used herein, "non-commercially"
#     means for research or evaluation purposes only.
#      3.4 Patent Claims. If you bring or threaten to bring a patent claim
#     against any Licensor (including any claim, cross-claim or
#     counterclaim in a lawsuit) to enforce any patents that you allege
#     are infringed by any Work, then your rights under this License from
#     such Licensor (including the grant in Section 2.1) will terminate
#     immediately.
#      3.5 Trademarks. This License does not grant any rights to use any
#     Licensor’s or its affiliates’ names, logos, or trademarks, except
#     as necessary to reproduce the notices described in this License.
#      3.6 Termination. If you violate any term of this License, then your
#     rights under this License (including the grant in Section 2.1) will
#     terminate immediately.
#  4. Disclaimer of Warranty.
#  THE WORK IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WARRANTIES OR CONDITIONS OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
# NON-INFRINGEMENT. YOU BEAR THE RISK OF UNDERTAKING ANY ACTIVITIES UNDER
# THIS LICENSE.
#  5. Limitation of Liability.
#  EXCEPT AS PROHIBITED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL
# THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), CONTRACT, OR OTHERWISE
# SHALL ANY LICENSOR BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT,
# INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES ARISING OUT OF
# OR RELATED TO THIS LICENSE, THE USE OR INABILITY TO USE THE WORK
# (INCLUDING BUT NOT LIMITED TO LOSS OF GOODWILL, BUSINESS INTERRUPTION,
# LOST PROFITS OR DATA, COMPUTER FAILURE OR MALFUNCTION, OR ANY OTHER
# COMMERCIAL DAMAGES OR LOSSES), EVEN IF THE LICENSOR HAS BEEN ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGES.
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import BatchNorm2d
from torch.nn import Module, Conv2d, Parameter
from mmcv.cnn import ConvModule
import fvcore.nn.weight_init as weight_init
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.ops import resize
import math

import torch
from torch import nn
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import cv2
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_color_map

from detectron2.layers import Conv2d, ShapeSpec, get_norm


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output
class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat
class PALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias), # channel <-> 1
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y
def l2_norm(x):
    return torch.einsum("bcn, bn->bcn", x, 1 / torch.norm(x, p=2, dim=-2))
class _HASPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_HASPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class HASP_1(nn.Module):
    def __init__(self, BatchNorm):
        """
        :param backbone: resnet
        :param output_stride: 16
        :param BatchNorm:
        """
        super(HASP_1, self).__init__()

        # HASP
        self.hasp1_1 = _HASPModule(128, 32, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.hasp1_2 = _HASPModule(128, 32, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.hasp2_1 = _HASPModule(128, 32, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.hasp2_2 = _HASPModule(128, 32, 3, padding=18, dilation=18, BatchNorm=BatchNorm)

        self._init_weight()

    def forward(self, x):
        x1_1 = self.hasp1_1(x)
        x1_2 = self.hasp1_2(x)
        x2_1 = self.hasp2_1(x)
        x2_2 = self.hasp2_2(x)


        x_all = torch.cat((x1_1,x1_2,x2_1,x2_2),dim=1)

        return x_all

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HASP_2(nn.Module):
    def __init__(self, BatchNorm):
        """
        :param backbone: resnet
        :param output_stride: 16
        :param BatchNorm:
        """
        super(HASP_2, self).__init__()

        # HASP
        self.hasp1_1 = _HASPModule(512, 128, 1, padding=0, dilation=1, BatchNorm=BatchNorm)
        self.hasp1_2 = _HASPModule(512, 128, 3, padding=6, dilation=6, BatchNorm=BatchNorm)
        self.hasp2_1 = _HASPModule(512, 128, 3, padding=12, dilation=12, BatchNorm=BatchNorm)
        self.hasp2_2 = _HASPModule(512, 128, 3, padding=18, dilation=18, BatchNorm=BatchNorm)

        self._init_weight()

    def forward(self, x):
        x1_1 = self.hasp1_1(x)
        x1_2 = self.hasp1_2(x)
        x2_1 = self.hasp2_1(x)
        x2_2 = self.hasp2_2(x)

        x_all = torch.cat((x1_1, x1_2, x2_1, x2_2), dim=1)

        return x_all

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class ASPPAttention_1(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(ASPPAttention_1, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.aspp = HASP_1(BatchNorm=nn.BatchNorm2d)
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        x_q = self.aspp(x)
        Q = self.query_conv(x_q).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()
class ASPPAttention_2(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(ASPPAttention_2, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps
        self.aspp = HASP_2(BatchNorm=nn.BatchNorm2d)
        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        # Apply the feature map to the queries and keys
        batch_size, chnnels, width, height = x.shape
        x_q = self.aspp(x)
        Q = self.query_conv(x_q).view(batch_size, -1, width * height)
        K = self.key_conv(x).view(batch_size, -1, width * height)
        V = self.value_conv(x).view(batch_size, -1, width * height)

        Q = self.l2_norm(Q).permute(-3, -1, -2)
        K = self.l2_norm(K)

        tailor_sum = 1 / (width * height + torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1) + self.eps))
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1)
        value_sum = value_sum.expand(-1, chnnels, width * height)

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(batch_size, chnnels, height, width)

        return (self.gamma * weight_value).contiguous()


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class FeatureAggregationModule_1(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureAggregationModule_1, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = ASPPAttention_1(out_chan)

        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)

        atten = self.conv_atten(feat)
        atten = torch.permute(atten,(0,1,3,2))

        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
class FeatureAggregationModule_2(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureAggregationModule_2, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv_atten = ASPPAttention_2(out_chan)

        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)

        atten = self.conv_atten(feat)
        atten = torch.permute(atten,(0,1,3,2))

        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=640,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)


        BatchNorm = nn.BatchNorm2d
        self.FSM_0 = FeatureSelectionModule(64, 64)
        self.FSM_1 = FeatureSelectionModule(128, 128)
        self.FSM_2 = FeatureSelectionModule(320, 320)
        self.FSM_3 = FeatureSelectionModule(512, 512)
        self.FAM_1 = FeatureAggregationModule_1(128, 128)
        self.FAM_2 = FeatureAggregationModule_2(512, 512)
        self.CAL_0 = PALayer(channel=64)
        self.CAL_1 = PALayer(channel=128)
        self.CAL_2 = PALayer(channel=320)
        self.CAL_3 = PALayer(channel=512)
        self.conv_1 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False),
                                    BatchNorm(64),
                                    nn.ReLU())
        self.conv_2 = nn.Sequential(nn.Conv2d(320, 256, 1, bias=False),
                                    BatchNorm(256),
                                    nn.ReLU())
        self.conv_3 = nn.Sequential(nn.Conv2d(512, 256, 1, bias=False),
                                    BatchNorm(256),
                                    nn.ReLU())
        self.conv_4 = nn.Sequential(nn.Conv2d(256, 128, 1, bias=False),
                                    BatchNorm(128),
                                    nn.ReLU())
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        # for idx in range(len(inputs)):
        #     x = inputs[idx]
        #     conv = self.convs[idx]
        #     outs.append(
        #         resize(
        #             input=conv(x),
        #             size=inputs[0].shape[2:],
        #             mode=self.interpolate_mode,
        #             align_corners=self.align_corners))
        feature_0 = self.FSM_0(inputs[0])
        feature_0 = self.CAL_0(feature_0)
        feature_1 = self.FSM_1(inputs[1])
        feature_1 = self.CAL_1(feature_1)
        # feature_0 = inputs[0]
        # feature_1 = inputs[1]
        # feature_2 = inputs[2]
        # feature_3 = inputs[3]
        feature_1 = F.interpolate(feature_1, size=feature_0.size()[2:], mode='bilinear', align_corners=True)

        feature_2 = self.FSM_2(inputs[2])
        feature_2 = self.CAL_2(feature_2)
        feature_2 = F.interpolate(feature_2, size=feature_0.size()[2:], mode='bilinear', align_corners=True)

        feature_3 = self.FSM_3(inputs[3])
        feature_3 = self.CAL_3(feature_3)


        feature_3 = F.interpolate(feature_3, size=feature_0.size()[2:], mode='bilinear', align_corners=True)
        feature_1 = self.conv_1(feature_1)
        feature_2 = self.conv_2(feature_2)
        feature_3 = self.conv_3(feature_3)


        # fam_1 = self.FAM_1(feature_0,feature_1)

        # fam_2 = self.FAM_2(feature_2,feature_3)


        # outs.append(fam_1)
        # outs.append(fam_2)
        outs.append(feature_0)
        outs.append(feature_1)
        outs.append(feature_2)
        outs.append(feature_3)



        out = self.fusion_conv(torch.cat(outs, dim=1))

        out = self.cls_seg(out)

        #可视化

        return out

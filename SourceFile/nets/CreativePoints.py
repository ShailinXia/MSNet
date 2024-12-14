#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：MSNet
@File    ：CreativePoints.py
@Author  ：Shailin Xia
@Date    ：2024/9/2 10:00
@Email   : shailinxia666@gmail.com
'''
from matplotlib import pyplot as plt
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
# from mmcv.cnn import kaiming_init
# from mmcv.cnn.bricks.context_block import last_zero_init
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn import functional as F


def autopad(k, p=None, d=1):
    # kernel, padding, dilation
    # 对输入的特征层进行自动padding，按照Same原则
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class ShuffleAttentionV1(nn.Module):
    def __init__(self, forward_state, channel=512, reduction=16, G=8, n_div=4):
        super(ShuffleAttentionV1, self).__init__()
        self.n_div = n_div
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.cbias = Parameter(torch.ones(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sweight = Parameter(torch.zeros(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sbias = Parameter(torch.ones(
            1, channel // (2 * G), 1, 1), requires_grad=True)
        self.sigmoid = nn.Sigmoid()

        if forward_state == 'forward_channel':
            self.forward = self.forward_channel
        elif forward_state == 'forward_spatial':
            self.forward = self.forward_spatial
        else:
            raise NotImplementedError

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)
        return x

    def forward_channel(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)
        x_0, _ = x.chunk(2, dim=1)

        # channel attention
        x_channel = self.avg_pool(x_0)
        x_channel = self.cweight * x_channel + self.cbias
        x_channel = x_0 * self.sigmoid(x_channel)
        return x_channel

    def forward_spatial(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)
        _, x_1 = x.chunk(2, dim=1)

        # spatial attention
        x_spatial = self.gn(x_1)
        x_spatial = self.sweight * x_spatial + self.sbias
        x_spatial = x_1 * self.sigmoid(x_spatial)
        return x_spatial


class PConv(nn.Module):
    def __init__(self, dim, ouc, n_div=4, type_forward='split_cat'):
        """
        Partial Convolution
        :param dim: Number of input channels
        :param ouc: Output channels
        :param n_div: Reciprocal of the partial ratio
        :param forward: Forward type, 'slicing' or 'split_cat'
        """
        super(PConv, self).__init__()
        self.type_forward = type_forward
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)
        self.conv = nn.Conv2d(dim, ouc, kernel_size=1)
        # self.conv = nn.Conv2d(dim, ouc, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)

        if type_forward == 'slicing':
            self.forward = self.forward_slicing
        elif type_forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(
            x[:, :self.dim_conv3, :, :])
        x = self.conv(x)
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # 将 x 通道分为两部分，一部分为 in_channel//n_div, 另一部分为 in_channel - in_channel//n_div
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        # 对 x1 部分卷积 --> in_channel = in_channel // n_div; output_channel = in_channel // n_div --> 保持通道数不变
        x1 = self.partial_conv3(x1)
        # 在通道上拼接 x1, x2
        x = torch.cat((x1, x2), 1)
        # 对 x 做一个卷积，改变通道数
        x = self.conv(x)
        return x


class PWConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1):
        super(PWConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k,
                              stride=s, groups=g, bias=False)

    def forward(self, x):
        return self.conv(x)


class CARAFE(nn.Module):
    def __init__(self, inC, outC, kernel_size=3, up_factor=2):
        super(CARAFE, self).__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(inC, inC // 4, 1)
        self.encoder = nn.Conv2d(inC // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(in_tensor)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.up_factor)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(
            2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(
            3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(
            N, self.kernel_size ** 2, H, W, self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(
            0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        in_tensor = F.pad(in_tensor, pad=(self.kernel_size // 2, self.kernel_size // 2,
                                          self.kernel_size // 2, self.kernel_size // 2),
                          mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        # (N, C, H, W+Kup//2+Kup//2, Kup)
        in_tensor = in_tensor.unfold(2, self.kernel_size, step=1)
        in_tensor = in_tensor.unfold(
            3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)  # (N, C, H, W, Kup^2)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        out_tensor = torch.matmul(
            in_tensor, kernel_tensor)  # (N, H, W, C, S^2)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.up_factor)
        out_tensor = self.out(out_tensor)
        return out_tensor


class P2CEB(nn.Module):
    """
    P2CEB --> Partial-Point-Channel-Extraction-Block
    """

    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(P2CEB, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.channel_shuffle = ShuffleAttentionV1.channel_shuffle
        self.pconv1 = PConv(c1, c_)
        self.spatial_attention = ShuffleAttentionV1(
            channel=c_, forward_state='forward_spatial')
        self.pwconv = PWConv(c1, c_, k=1, g=g)
        self.channel_attention = ShuffleAttentionV1(
            channel=c_, forward_state='forward_channel')
        self.bn = nn.BatchNorm2d(c_)
        self.relu = nn.ReLU(inplace=True)
        self.pconv2 = PConv(c_, c2)
        self.add = shortcut and c1 == c2
        self.ln = nn.LayerNorm([c_, 1, 1])

    def forward(self, x):
        b, c, h, w = x.size()

        x_spatial = self.spatial_attention(self.pconv1(x))
        x_channel = self.channel_attention(self.pwconv(x))
        # concatenate features
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        out = self.channel_shuffle(out, 2)
        # BN + ReLU
        out = self.relu(self.bn(out))
        # out = self.relu(self.ln(out))

        if self.add:
            out = x + self.pconv2(out)
        else:
            out = self.pconv2(out)
        return out


class GlobalContextBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalContextBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.Conv_key = nn.Conv2d(self.in_channels, 1, 1)
        self.SoftMax = nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            # nn.Conv2d(self.in_channels, self.out_channels, 1),
            PWConv(self.in_channels, self.out_channels, 1),
            nn.LayerNorm([self.out_channels, 1, 1]),
            nn.ReLU(),
            # nn.Conv2d(self.out_channels, self.in_channels, 1),
            PWConv(self.out_channels, self.in_channels, 1),
        )
        self.change_channels = PWConv(self.in_channels, self.out_channels, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        # key -> [b, 1, H, W] -> [b, 1, H*W] ->  [b, H*W, 1]
        # torch.contiguous() 类似于深拷贝 deep copy
        key = self.SoftMax(self.Conv_key(x).view(
            b, 1, -1).permute(0, 2, 1).view(b, -1, 1).contiguous())
        query = x.view(b, c, h * w)
        # [b, c, h*w] matmul [b, h*w, 1] --> b, c, 1, 1
        concate_QK = torch.matmul(query, key)
        concate_QK = concate_QK.view(b, c, 1, 1).contiguous()
        value = self.Conv_value(concate_QK)
        out = self.change_channels(x + value)
        # out = x + value
        return out


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


class SiLU(nn.Module):
    # SiLU激活函数
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


class Conv(nn.Module):
    # 标准卷积+标准化+激活函数
    default_act = SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(
            c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act = self.default_act if act is True else act if isinstance(
            act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # 标准瓶颈结构，残差结构
    # c1为输入通道数，c2为输出通道数
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, n=1, e=0.5):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes

        self.c = int(out_planes * e)
        self.cv1 = PWConv(in_planes, 2 * self.c, 1, 1)
        self.cv2 = PWConv((2 + n) * self.c, out_planes, 1)
        self.m = nn.ModuleList(Bottleneck(
            self.out_planes // 2, self.out_planes // 2) for _ in range(n))

        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))

        self.conv = nn.Conv2d(self.in_planes, self.out_planes, 3, padding=1)
        self.conv_key = PWConv(in_planes, 1)

        self.softmax = torch.nn.Softmax(dim=1)

        self.Conv_value = nn.Sequential(
            PWConv(self.in_planes, self.out_planes, 1),
            nn.LayerNorm([self.out_planes, 1, 1]),
            nn.ReLU(),
            PWConv(self.out_planes, self.in_planes, 1),
        )
        self.change_channels = PWConv(self.in_planes, self.out_planes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

    def forward(self, x):
        k_trans = self.conv_key(x)
        b, c, h, w = x.shape

        key = self.softmax(k_trans.view(
            b, 1, -1).permute(0, 2, 1).contiguous())
        query = x.view(b, -1, h * w)
        # b, c, h*w matmul b, h*w, 1 --> b, c, 1
        concate_QK = torch.matmul(query, key).view(b, c, 1, 1).contiguous()
        out_att = self.change_channels(x + self.Conv_value(concate_QK))

        # conv
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        out_conv = self.cv2(torch.cat(y, 1))

        return self.rate1 * out_att + self.rate2 * out_conv

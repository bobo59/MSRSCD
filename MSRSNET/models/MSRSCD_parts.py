import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import OrderedDict
import math

bn_mom = 0.0003
"""Implemention of non-local feature pyramid network"""


class NL_Block(nn.Module):
    def __init__(self, in_channels):
        super(NL_Block, self).__init__()
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        batch_size, c, h, w = x.size(0), x.size(1), x.size(2), x.size(3)
        value = self.conv_v(x).view(batch_size, c, -1)
        value = value.permute(0, 2, 1)  # B * (H*W) * value_channels
        key = x.view(batch_size, c, -1)  # B * key_channels * (H*W)
        query = x.view(batch_size, c, -1)
        query = query.permute(0, 2, 1)
        sim_map = torch.matmul(query, key)  # B * (H*W) * (H*W)
        sim_map = (c ** -.5) * sim_map  # B * (H*W) * (H*W)
        sim_map = torch.softmax(sim_map, dim=-1)  # B * (H*W) * (H*W)
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, c, *x.size()[2:])
        context = self.W(context)

        return context


class NL_FPN(nn.Module):
    """ non-local feature parymid network"""

    def __init__(self, in_dim, reduction=True):
        super(NL_FPN, self).__init__()
        if reduction:
            self.reduction = nn.Sequential(
                nn.Conv2d(in_dim, in_dim // 4, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.re_reduction = nn.Sequential(
                nn.Conv2d(in_dim // 4, in_dim, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
            )
            in_dim = in_dim // 4
        else:
            self.reduction = None
            self.re_reduction = None
        self.conv_e1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_e2 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.conv_e3 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 4),
            nn.ReLU(inplace=True),
        )
        self.conv_d1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_d2 = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
        )
        self.conv_d3 = nn.Sequential(
            nn.Conv2d(in_dim * 4, in_dim * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_dim * 2),
            nn.ReLU(inplace=True),
        )
        self.nl3 = NL_Block(in_dim * 2)
        self.nl2 = NL_Block(in_dim)
        self.nl1 = NL_Block(in_dim)

        self.downsample_x2 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.upsample_x2 = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        if self.reduction is not None:
            x = self.reduction(x)
        e1 = self.conv_e1(x)  # C,H,W
        e2 = self.conv_e2(self.downsample_x2(e1))  # 2C,H/2,W/2
        e3 = self.conv_e3(self.downsample_x2(e2))  # 4C,H/4,W/4

        d3 = self.conv_d3(e3)  # 2C,H/4,W/4
        nl = self.nl3(d3)
        d3 = self.upsample_x2(torch.mul(d3, nl))  ##2C,H/2,W/2
        d2 = self.conv_d2(e2 + d3)  # C,H/2,W/2
        nl = self.nl2(d2)
        d2 = self.upsample_x2(torch.mul(d2, nl))  # C,H,W
        d1 = self.conv_d1(e1 + d2)
        nl = self.nl1(d1)
        d1 = torch.mul(d1, nl)  # C,H,W
        if self.re_reduction is not None:
            d1 = self.re_reduction(d1)

        return d1


class double_conv(torch.nn.Module):
    def __init__(self, in_chn, out_chn, stride=1,
                 dilation=1):  # params:in_chn(input channel of double conv),out_chn(output channel of double conv)
        super(double_conv, self).__init__()  ##parent's init func

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=stride, dilation=dilation, padding=dilation),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_chn, out_chn, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class SKAttention(nn.Module):
    def __init__(self, channel=512):
        super().__init__()
        self.d = max(32, channel // 16)
        self.convs = nn.ModuleList([])
        for k in [1, 3, 5, 7]:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=1)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        self.fc = nn.Linear(channel, self.d)
        self.fcs = nn.ModuleList([])
        for _ in range(len([1, 3, 5, 7])):
            self.fcs.append(nn.Linear(self.d, channel))
        self.softmax = nn.Softmax(dim=0)
        self.ca = ChannelAttention(channel, 16)
        self.sa = SpatialAttention(3)

    def forward(self, x):
        conv_outs = [conv(x) for conv in self.convs]
        feats = torch.stack(conv_outs, dim=0)
        U = torch.sum(feats, dim=0)
        out = U * self.ca(x)
        result = out * self.sa(out)
        return result

class Encoder(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(Encoder, self).__init__()
        self.conv1 = double_conv(in_chn, out_chn)
        self.conv2 = double_conv(out_chn, out_chn, stride=2)
        self.attention = SKAttention(out_chn)
        self.downsample = torch.nn.MaxPool2d(stride=2, kernel_size=2)
        self.ReLU = nn.ReLU(inplace=True)
    # def forward(self, x):
    #     x = self.conv1(x)
    #     residual = x
    #     x = self.conv2(x)
    #     x = self.attention(x)
    #     residual = self.downsample(residual)
    #     x = x + residual
    #     x = self.ReLU(x)
    #     return x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = x
        x = self.attention(x)
        x = x + residual
        x = self.ReLU(x)
        return x


class SEModule(nn.Module):

    def __init__(self, channels, reduction_channels):
        super(SEModule, self).__init__()
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, reduction_channels, kernel_size=1, padding=0, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            reduction_channels, channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        #x_se = self.avg_pool(x)
        x_se = x.view(x.size(0), x.size(1), -1).mean(-1).view(x.size(0), x.size(1), 1, 1)
        x_se = self.fc1(x_se)
        x_se = self.ReLU(x_se)
        x_se = self.fc2(x_se)
        return x * x_se.sigmoid()

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, downsample=None, use_se=False, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        first_planes = planes
        outplanes = planes * self.expansion

        self.conv1 = double_conv(inplanes, first_planes)
        self.conv2 = double_conv(first_planes, outplanes, stride=stride, dilation=dilation)
        self.se = SEModule(outplanes, planes // 4) if use_se else None
        self.downsample = torch.nn.MaxPool2d(stride=2,kernel_size=2) if downsample else None
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv1(x)
        residual = out
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out = out + residual
        out = self.ReLU(out)

        return out

class BasicConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size, stride, padding=1):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_chn, out_chn, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_chn)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class cat(torch.nn.Module):
    def __init__(self, in_chn_high, in_chn_low, out_chn, upsample=False):
        super(cat, self).__init__()  ##parent's init func
        self.do_upsample = upsample
        self.upsample = torch.nn.Upsample(
            scale_factor=2, mode="nearest"
        )
        self.conv2d = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn_high + in_chn_low, out_chn, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        # import ipdb
        # ipdb.set_trace()
        if self.do_upsample:
            x = self.upsample(x)

        x = torch.cat((x, y), 1)  # x,y shape(batch_sizxe,channel,w,h), concat at the dim of channel
        return self.conv2d(x)

class densecat_cat_add(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_add, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)

class densecat_cat_diff(nn.Module):
    def __init__(self, in_chn, out_chn):
        super(densecat_cat_diff, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2+x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2+y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out

class DF_Module(nn.Module):
    def __init__(self, dim_in, dim_out, reduction=True):
        super(DF_Module, self).__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in//2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in//2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in//2
        else:
            self.reduction = None
        self.cat1 = densecat_cat_add(dim_in, dim_out)
        self.cat2 = densecat_cat_diff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + x_add
        return y


# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=512):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,in_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, in_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(in_channel * 5, in_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = nn.Sequential(
            nn.Conv2d(c1, c_, 1, 1),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )
        self.cv2 = nn.Sequential(
            nn.Conv2d(c_ *4, c2, 1, 1),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class GMSFF(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(c,c,3,1,1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(c,c,5,1,2),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(2*c,c,1,1),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True)
        )

    def forward(self,x):
        x1=self.conv1(x)
        x2=self.conv2(x)
        x3=torch.cat([x1,x2],1)
        x4=self.conv3(x3)
        return x+x4

def kernel_size(in_channel):
    """Compute kernel size for one dimension convolution in eca-net"""
    k = int((math.log2(in_channel) + 1) // 2)  # parameters from ECA-net
    if k % 2 == 0:
        return k + 1
    else:
        return k
class CBAM(nn.Module):
    """Attention module."""

    def __init__(self, in_channel):
        super().__init__()
        self.k = kernel_size(in_channel)
        self.channel_conv = nn.Conv1d(2, 1, kernel_size=self.k, padding=self.k // 2)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg_channel = self.avg_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        max_channel = self.max_pooling(x).squeeze(-1).transpose(1, 2)  # b,1,c
        channel_weight = self.channel_conv(torch.cat([avg_channel, max_channel], dim=1))
        channel_weight = self.sigmoid(channel_weight).transpose(1, 2).unsqueeze(-1)  # b,c,1,1
        x = channel_weight * x

        avg_spatial = torch.mean(x, dim=1, keepdim=True)  # b,1,h,w
        max_spatial = torch.max(x, dim=1, keepdim=True)[0]  # b,1,h,w
        spatial_weight = self.spatial_conv(torch.cat([avg_spatial, max_spatial], dim=1))  # b,1,h,w
        spatial_weight = self.sigmoid(spatial_weight)
        output = spatial_weight * x
        return output


class oneConv(nn.Module):
    # 卷积+ReLU函数
    def __init__(self, in_channels, out_channels, kernel_sizes, paddings, dilations):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//2, kernel_size = kernel_sizes, padding = paddings, dilation = dilations),
            nn.Conv2d(out_channels//2, out_channels, kernel_size = kernel_sizes, padding = paddings, dilation = dilations),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class MSFblock(nn.Module):
    def __init__(self, in_channels):
        super(MSFblock, self).__init__()
        out_channels = in_channels
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = oneConv(in_channels,in_channels,1,0,1)
        self.SE2 = oneConv(in_channels,in_channels,1,0,1)
        self.SE3 = oneConv(in_channels,in_channels,1,0,1)
        self.SE4 = oneConv(in_channels,in_channels,1,0,1)


    def forward(self, x0,x1,x2,x3):
        # x1/x2/x3/x4: (B,C,H,W)
        y0 = x0
        y1 = x1
        y2 = x2
        y3 = x3

        # 通过池化聚合全局信息,然后通过1×1conv建模通道相关性: (B,C,H,W)-->GAP-->(B,C,1,1)-->SE1-->(B,C,1,1)
        y0_weight = self.SE1(self.gap(x0))
        y1_weight = self.SE2(self.gap(x1))
        y2_weight = self.SE3(self.gap(x2))
        y3_weight = self.SE4(self.gap(x3))

        # 将多个尺度的全局信息进行拼接: (B,C,4,1)
        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)
        # 首先通过sigmoid函数获得通道描述符表示, 然后通过softmax函数,求每个尺度的权重: (B,C,4,1)--> (B,C,4,1)
        weight = self.softmax(self.Sigmoid(weight))

        # weight[:,:,0]:(B,C,1); (B,C,1)-->unsqueeze-->(B,C,1,1)
        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)

        # 将权重与对应的输入进行逐元素乘法: (B,C,1,1) * (B,C,H,W)= (B,C,H,W), 然后将多个尺度的输出进行相加
        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.project(x_att)

class ChangerChannelExchangeAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.p = 2
        self.MSFblock = MSFblock(in_channels)  # Ensure MSFblock is defined elsewhere

    def forward(self, x1, x2):
        device = x1.device  # Determine the device from the input tensors
        N, C, H, W = x1.shape

        # Create a boolean exchange mask; no need to cast to int or manually expand
        exchange_mask = (torch.arange(C, device=device) % self.p == 0)

        # Directly use boolean masks with broadcasting
        y1 = x1 * exchange_mask[:, None, None]
        y2 = x2 * (~exchange_mask)[:, None, None]
        y3 = x1 * (~exchange_mask)[:, None, None]
        y4 = x2 * exchange_mask[:, None, None]

        # Pass the masked inputs to the MSFblock
        out = self.MSFblock(y1, y2, y3, y4)

        return out

class DiFF(nn.Module):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64):
        super(DiFF, self).__init__()
        r = 4
        inter_channels = int(channels // r)

        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(channels),
        )

        # 第二次本地注意力
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        # 第二次全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        return xo

if __name__=='__main__':
    x1 = torch.randn(2, 16, 32, 32)
    x2 = torch.randn(2, 16, 32, 32)
    model = ChangerChannelExchangeAttention(16, 16)
    out = model(x1, x2)

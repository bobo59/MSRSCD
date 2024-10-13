import torch
import torch.nn as nn


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
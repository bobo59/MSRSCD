import torch
import torch.nn as nn
from models.MSRSCD_parts import BasicBlock, NL_FPN, DF_Module, cat,ASPP,SPPF,GMSFF,ChangerChannelExchangeAttention,Encoder,DiFF

from models.dpcd_parts import Changer_channel_exchange,DPFA

class MSRSCD(nn.Module):
    def __init__(self):
        super(MSRSCD, self).__init__()

        dilation_list = [1, 1, 1, 1]
        stride_list = [2, 2, 2, 2]
        pool_list = [True, True, True, True]

        se_list = [True, True, True, True]
        channel_list = [256, 128, 64, 32]
        # encoder
        # self.block1 = BasicBlock(3, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        # self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        # self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        # self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])
        self.block1 = Encoder(3, channel_list[3])
        self.block2 = Encoder(channel_list[3], channel_list[2])
        self.block3 = Encoder(channel_list[2], channel_list[1])
        self.block4 = Encoder(channel_list[1], channel_list[0])

        # center
        # self.center = NL_FPN(channel_list[0], True)
        self.center = GMSFF(channel_list[0])


        # decoder
        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        # self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        # self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        # self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        # self.df4 = DF_Module(channel_list[0], channel_list[0], True)
        self.df1 = ChangerChannelExchangeAttention(channel_list[3])
        self.df2 = ChangerChannelExchangeAttention(channel_list[2])
        self.df3 = ChangerChannelExchangeAttention(channel_list[1])
        self.df4 = ChangerChannelExchangeAttention(channel_list[0])
        # self.df1 = DPFA(channel_list[3])
        # self.df2 = DPFA(channel_list[2])
        # self.df3 = DPFA(channel_list[1])
        # self.df4 = DPFA(channel_list[0])

        self.catc3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(8),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8 ,1 ,kernel_size=3 ,stride=1 ,padding=1)
        self.conv_out_class = torch.nn.Conv2d(channel_list[3] ,1, kernel_size=1 ,stride=1 ,padding=0)


        self.channel_exchange4 = Changer_channel_exchange()


    def forward(self, x1, x2):
        # e1_1 = self.block1(x[0])
        e1_1 = self.block1(x1)
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        # e1_2 = self.block1(x[1])
        e1_2 = self.block1(x2)
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        # y1, y2 = self.channel_exchange4(y1, y2)

        y1 = self.center(y1)
        y2 = self.center(y2)
        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        seg_out1 = self.conv_out_class(y1)
        seg_out2 = self.conv_out_class(y2)
        change_out = self.conv_out(self.upsample_x2(c))
        return change_out, seg_out1, seg_out2

if __name__ == "__main__":
    from torchsummary import summary
    from thop import profile

    model = MSRSCD()
    summary(model, input_size=[(3, 1024, 1024), (3, 1024, 1024)], batch_size=1, device="cpu")
    flops, params = profile(model, inputs=(torch.randn(1, 3, 1024, 1024), torch.randn(1, 3, 1024, 1024)))
    print(f"FLOPs: {flops}, Params: {params}")
import torch
import torch.nn as nn
from .PMNet import PMNet
from .AreaAtten import MyABlock
from .CrossAreaAttn import CrossAttn

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.LeakyReLU()
    )


def channel_tune(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
    )


def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=True),
        nn.LeakyReLU(),
        nn.Conv2d(ch_fine, ch_fine, 3, 1, 1, bias=True),
        nn.LeakyReLU()
    )

# fusion network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.max_pool = nn.MaxPool2d(2)

        self.init_conv_vis = channel_tune(1, 16, 1, 1, 0)

        self.ir_extract_fea1 = PMNet(1, 16)
        self.ir_extract_fea2 = PMNet(16, 32)
        self.ir_extract_fea3 = PMNet(32, 64)
        self.ir_extract_fea4 = PMNet(64, 128)

        self.interaction1 = CrossAttn(16, 1, 8)
        self.interaction2 = CrossAttn(32, 1, 8)
        self.interaction3 = CrossAttn(64, 2, 4)
        self.interaction4 = CrossAttn(128, 4, 4)


        self.add_channel1 = channel_tune(16, 32)
        self.add_channel2 = channel_tune(32, 64)
        self.add_channel3 = channel_tune(64, 128)

        self.ABlock1 = MyABlock(16, 1, 8)
        self.ABlock2 = MyABlock(32, 1, 8)
        self.ABlock3 = MyABlock(64, 2, 4)
        self.ABlock4 = MyABlock(128, 4, 4)

        self.inject1 = channel_tune(32, 16)
        self.inject2 = channel_tune(64, 32)
        self.inject3 = channel_tune(128, 64)
        self.inject4 = channel_tune(256, 128)

        self.upsample1 = upsample(128, 64)
        self.upsample2 = upsample(64, 32)
        self.upsample3 = upsample(32, 16)

        self.conv_out1 = add_conv_stage(128, 64)
        self.conv_out2 = add_conv_stage(64, 32)
        self.conv_out3 = add_conv_stage(32, 16)
        self.conv0m = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1), nn.Sigmoid())


    def forward(self, img_ir, img_vis):

        img_vis = self.init_conv_vis(img_vis)

        img_ir1 = self.ir_extract_fea1(img_ir)
        img_vis1 = self.ABlock1(img_vis)
        ir_vis1 = self.interaction1(img_ir1,img_vis1)
        img_vis1_out = self.inject1(torch.cat((img_vis1, ir_vis1), dim = 1))
        ir_vis1 = img_vis1_out

        img_ir2 = self.ir_extract_fea2(self.max_pool(img_ir1))
        img_vis1_out = self.add_channel1(self.max_pool(img_vis1_out))
        img_vis2_out = self.ABlock2(img_vis1_out)
        ir_vis2 = self.interaction2(img_ir2, img_vis2_out)
        img_vis2_out = self.inject2(torch.cat((img_vis2_out, ir_vis2), dim = 1))
        ir_vis2 = img_vis2_out

        img_ir3 = self.ir_extract_fea3(self.max_pool(img_ir2))
        img_vis2_out = self.add_channel2(self.max_pool(img_vis2_out))
        img_vis3_out = self.ABlock3(img_vis2_out)
        ir_vis3 = self.interaction3(img_ir3, img_vis3_out)
        img_vis3_out = self.inject3(torch.cat((img_vis3_out, ir_vis3), dim = 1))
        ir_vis3 = img_vis3_out

        img_ir4 = self.ir_extract_fea4(self.max_pool(img_ir3))
        img_vis3_out = self.add_channel3(self.max_pool(img_vis3_out))
        img_vis4_out = self.ABlock4(img_vis3_out)
        ir_vis4 = self.interaction4(img_ir4, img_vis4_out)
        img_vis4_out = self.inject4(torch.cat((img_vis4_out, ir_vis4), dim = 1))
        ir_vis4 = img_vis4_out

        output1 = self.upsample1(ir_vis4)
        output1 = torch.cat((output1, ir_vis3), dim=1)
        output1 = self.conv_out1(output1)

        output2 = self.upsample2(output1)
        output2 = torch.cat((output2, ir_vis2), dim=1)
        output2 = self.conv_out2(output2)

        output3 = self.upsample3(output2)
        output3 = torch.cat((output3, ir_vis1), dim=1)
        output3 = self.conv_out3(output3)

        output = self.conv0m(output3)

        return output
import math
import torch
import torch.nn as nn
#from .utils import unetConv2, unetUp
#from utils import util
from models.networks.utils import unetConv2,unetUp,UnetDsv2
from models.layers.nonlocal_layer import NONLocalBlock2D
import torch.nn.functional as F
#from .ASFFmobile import ASFF
from models.networks.ASFFmobile import ASFF

class unet_nonlocal_dsv_2D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 is_batchnorm=True, nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_dsv_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal1 = NONLocalBlock2D(in_channels=filters[0], inter_channels=filters[0] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal2 = NONLocalBlock2D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4]+filters[3], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1]+filters[0], filters[0], self.is_deconv)
        
        self.dsv4 = UnetDsv2(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv2(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv2(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)
        
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.final_2 = nn.Conv2d(n_classes*4, n_classes, 1)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        nonlocal1 = self.nonlocal1(maxpool1)

        conv2 = self.conv2(nonlocal1)
        maxpool2 = self.maxpool2(conv2)
        nonlocal2 = self.nonlocal2(maxpool2)

        conv3 = self.conv3(nonlocal2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        
        #final = self.final(up1)
        
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final_2(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))
        
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p

class unet_nonlocal_2D(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 is_batchnorm=True, nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_2D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        print("filters",filters)
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #print("model",nonlocal_mode)
        self.nonlocal1 = NONLocalBlock2D(in_channels=filters[0], inter_channels=filters[0] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal2 = NONLocalBlock2D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling      #
        self.up_concat4 = unetUp(filters[4]+filters[3], filters[3], self.is_deconv) #256+128->128
        self.up_concat3 = unetUp(filters[3]+filters[2], filters[2], self.is_deconv) #128+64 ->64
        self.up_concat2 = unetUp(filters[2]+filters[1], filters[1], self.is_deconv) #64+32  ->32
        self.up_concat1 = unetUp(filters[1]+filters[0], filters[0], self.is_deconv) #32+16  ->16

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.s=nn.Sigmoid()
        self.soft=nn.Softmax()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):                     #16, 32, 64, 128, 256
        conv1 = self.conv1(inputs)             #14, 16, 256, 256
        maxpool1 = self.maxpool1(conv1)        #14, 16, 128, 128
        nonlocal1 = self.nonlocal1(maxpool1)   #14, 16, 128, 128

        #print("conv1",conv1.shape)
        #print("maxpool1", maxpool1.shape)
        #print("nonlocal1", nonlocal1.shape)

        conv2 = self.conv2(nonlocal1)         #14, 32, 128, 128
        maxpool2 = self.maxpool2(conv2)       #14, 32, 64, 64
        nonlocal2 = self.nonlocal2(maxpool2)  #14, 32, 64, 64
        #print("conv2", conv2.shape)
        #print("maxpool2", maxpool2.shape)
        #print("nonlocal2", nonlocal2.shape)

        conv3 = self.conv3(nonlocal2)         #14, 64, 64, 64
        maxpool3 = self.maxpool3(conv3)       #14, 64, 32, 32
        #print("conv3", conv3.shape)
        #print("maxpool3", maxpool3.shape)

        conv4 = self.conv4(maxpool3)          #14, 128, 32, 32
        maxpool4 = self.maxpool4(conv4)       #14, 128, 16, 16
        #print("conv4", conv4.shape)
        #print("maxpool4", maxpool4.shape)

        center = self.center(maxpool4)        #14, 256, 16, 16
        #print("center", center.shape)
        up4 = self.up_concat4(conv4, center)  #14, 128, 32, 32+14, 256, 16, 16->14, 128, 32, 32
        #print("up4", up4.shape)

        up3 = self.up_concat3(conv3, up4)     #14, 64, 64, 64
        #print("up3", up3.shape)

        up2 = self.up_concat2(conv2, up3)     #14, 32, 128, 128
        #print("up2", up2.shape)

        up1 = self.up_concat1(conv1, up2)     #14, 16, 256, 256
        #print("up1", up1.shape)

        final = self.final(up1)               #14, 2, 256, 256
        #print("final", final.shape)
        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p

class unet_nonlocal_2D_ASFF(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 is_batchnorm=True, nonlocal_mode='embedded_gaussian', nonlocal_sf=4):
        super(unet_nonlocal_2D_ASFF, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        print("filters",filters)
        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        #print("model",nonlocal_mode)
        self.nonlocal1 = NONLocalBlock2D(in_channels=filters[0], inter_channels=filters[0] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.nonlocal2 = NONLocalBlock2D(in_channels=filters[1], inter_channels=filters[1] // 4,
                                         sub_sample_factor=nonlocal_sf, mode=nonlocal_mode)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling      #
        self.up_concat4 = unetUp(filters[4]+filters[3], filters[3], self.is_deconv) #256+128->128
        self.up_concat3 = unetUp(filters[3]+filters[2], filters[2], self.is_deconv) #128+64 ->64
        self.up_concat2 = unetUp(filters[2]+filters[1], filters[1], self.is_deconv) #64+32  ->32
        self.up_concat1 = unetUp(filters[1]+filters[0], filters[0], self.is_deconv) #32+16  ->16

        #self.ASFF_0 = ASFF(level=0, rfb=True,dim=[64, 32, 16])
        #self.ASFF_1 = ASFF(level=1, rfb=True,dim=[64, 32, 16])
        self.ASFF_2 = ASFF(level=2, rfb=True,dim=[64, 32, 16])
        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)
        self.final_A = nn.Conv2d(32, n_classes, 1)

        self.s=nn.Sigmoid()
        self.soft=nn.Softmax()

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, inputs):                     #16, 32, 64, 128, 256
        conv1 = self.conv1(inputs)             #14, 16, 256, 256
        maxpool1 = self.maxpool1(conv1)        #14, 16, 128, 128
        nonlocal1 = self.nonlocal1(maxpool1)   #14, 16, 128, 128

        #print("conv1",conv1.shape)
        #print("maxpool1", maxpool1.shape)
        #print("nonlocal1", nonlocal1.shape)

        conv2 = self.conv2(nonlocal1)         #14, 32, 128, 128
        maxpool2 = self.maxpool2(conv2)       #14, 32, 64, 64
        nonlocal2 = self.nonlocal2(maxpool2)  #14, 32, 64, 64
        #print("conv2", conv2.shape)
        #print("maxpool2", maxpool2.shape)
        #print("nonlocal2", nonlocal2.shape)

        conv3 = self.conv3(nonlocal2)         #14, 64, 64, 64
        maxpool3 = self.maxpool3(conv3)       #14, 64, 32, 32
        #print("conv3", conv3.shape)
        #print("maxpool3", maxpool3.shape)

        conv4 = self.conv4(maxpool3)          #14, 128, 32, 32
        maxpool4 = self.maxpool4(conv4)       #14, 128, 16, 16
        #print("conv4", conv4.shape)
        #print("maxpool4", maxpool4.shape)

        center = self.center(maxpool4)        #14, 256, 16, 16
        #print("center", center.shape)
        up4 = self.up_concat4(conv4, center)  #14, 128, 32, 32+14, 256, 16, 16->14, 128, 32, 32
        #print("up4", up4.shape)

        up3 = self.up_concat3(conv3, up4)     #14, 64, 64, 64
        #print("up3", up3.shape)

        up2 = self.up_concat2(conv2, up3)     #14, 32, 128, 128
        #print("up2", up2.shape)

        up1 = self.up_concat1(conv1, up2)     #14, 16, 256, 256
        #print("up1", up1.shape)

        #A_0 = self.ASFF_0(up3, up2, up1)        #14, 128, 64, 64
        #A_1 = self.ASFF_1(up3, up2, up1)        #14, 64, 128, 128
        A_2 = self.ASFF_2(up3, up2, up1)         #14, 32, 256, 256

        #print("A_2",A_2.shape)
        #final = self.final(up1)                 #14, 2, 256, 256
        final_A = self.final_A(A_2)

        #print("final", final.shape)
        #return final,final_A
        return final_A

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p



def main(net_name="unet"):

    net_name="unet"
    png_name = "4_0375.png"
    png_name = "4_0398.png"   #0 0
    x=torch.rand(1,1,512,512)
    U_models=unet_nonlocal_2D_ASFF(n_classes=2,
                              is_batchnorm=True,
                              in_channels=1,
                              is_deconv=False,
                              nonlocal_mode="embedded_gaussian",
                              feature_scale=4)

    y=U_models(x)
    print("out shape",y.shape)

if __name__=='__main__':

    main()

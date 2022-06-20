import torch.nn as nn
import torch
from torch.autograd import Variable
#from utils import unetConv2, UnetUp2_CT, UnetGridGatingSignal2, UnetDsv2
from models.networks.utils import unetConv2, UnetUp2_CT, UnetGridGatingSignal2, UnetDsv2
import torch.nn.functional as F
from models.networks_other import init_weights
#from ../networks_other import init_weights
from models.layers.grid_attention_layer import GridAttentionBlock2D
import cv2 as cv
class unet_CT_2D_add_pic(nn.Module):
    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3,
                 nonlocal_mode='concatenation', attention_dsample=(2,2), is_batchnorm=True):
        super(unet_CT_2D_add_pic, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        nonlocal_mode = 'concatenation'
        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, ks=(3,3), padding=(1,1))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,  2))

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, ks=(3,3), padding=(1,1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, ks=(3,3), padding=(1,1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,  2))

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, ks=(3,3), padding=(1,1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 2))

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, ks=(3,3), padding=(1,1))
        self.gating = UnetGridGatingSignal2(filters[4], filters[4], kernel_size=(1, 1), is_batchnorm=self.is_batchnorm)

        self.conv1_add = unetConv2(1, 32, self.is_batchnorm, ks=(3,3), padding=(1,1))
        #self.maxpool_1_add = nn.MaxPool2d(kernel_size=(2, 2))
        
        self.conv2_add = unetConv2(1, 64, self.is_batchnorm, ks=(3,3), padding=(1,1))
        #self.maxpool_2_add = nn.MaxPool2d(kernel_size=(2, 2))
        # attention blocks
        self.attentionblock2 = MultiAttentionBlock(in_size=filters[1], gate_size=filters[2], inter_size=filters[1],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock3 = MultiAttentionBlock(in_size=filters[2], gate_size=filters[3], inter_size=filters[2],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)
        self.attentionblock4 = MultiAttentionBlock(in_size=filters[3], gate_size=filters[4], inter_size=filters[3],
                                                   nonlocal_mode=nonlocal_mode, sub_sample_factor= attention_dsample)

        # upsampling
        self.up_concat4 = UnetUp2_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp2_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp2_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp2_CT(filters[1], filters[0], is_batchnorm)

        # deep supervision
        self.dsv4 = UnetDsv2(in_size=filters[3], out_size=n_classes, scale_factor=8)
        self.dsv3 = UnetDsv2(in_size=filters[2], out_size=n_classes, scale_factor=4)
        self.dsv2 = UnetDsv2(in_size=filters[1], out_size=n_classes, scale_factor=2)
        self.dsv1 = nn.Conv2d(in_channels=filters[0], out_channels=n_classes, kernel_size=1)

        # final conv (without any concat)
        self.final = nn.Conv2d(n_classes*4, n_classes, 1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # Feature Extraction
        #inputs_ore=inputs.numpy()
        #print("input shape",inputs_ore.shape)
        input_2_x=inputs.squeeze(1).permute(1,2,0).cpu().detach().numpy()
        input_2_y=cv.resize(input_2_x,(inputs.size(2)//4,inputs.size(2)//4))
        input_3_y=cv.resize(input_2_x,(inputs.size(2)//8,inputs.size(2)//8))
        input_2=torch.from_numpy(input_2_y).permute(2,0,1).unsqueeze(1)
        input_3=torch.from_numpy(input_3_y).permute(2,0,1).unsqueeze(1)

        conv1_add=self.conv1_add(Variable(input_2.cuda()))
        conv2_add=self.conv2_add(Variable(input_3.cuda()))
        #print("input shape:",inputs.shape)
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        #print("input shape:",inputs.shape)
        #print("conv shape:",conv1.shape)
        #print("lay1 out:",maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        maxpool2=maxpool2+conv1_add
        #print("lay2 out:",maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        maxpool3=maxpool3+conv2_add
        #print("lay3 out:",maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        #print("lay4 out:",maxpool4.shape)

        # Gating Signal Generation
        center = self.center(maxpool4)
        gating = self.gating(center)

        # Attention Mechanism
        # Upscaling Part (Decoder)
        g_conv4, att4 = self.attentionblock4(conv4, gating)
        up4 = self.up_concat4(g_conv4, center)
        
        g_conv3, att3 = self.attentionblock3(conv3, up4)
        up3 = self.up_concat3(g_conv3, up4)
        
        g_conv2, att2 = self.attentionblock2(conv2, up3)
        up2 = self.up_concat2(g_conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # Deep Supervision
        dsv4 = self.dsv4(up4)
        dsv3 = self.dsv3(up3)
        dsv2 = self.dsv2(up2)
        dsv1 = self.dsv1(up1)
        final = self.final(torch.cat([dsv1,dsv2,dsv3,dsv4], dim=1))

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)

        return self.combine_gates(gate_1), attention_1

if __name__=='__main__':
    x=torch.rand(1,1,256,256)
    model=unet_CT_att_dsv_2D(feature_scale=4,n_classes=3,in_channels=1)
    sip=model(x)
    print("out shape",sip.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F
#14,16,256,256   14,32,128,128    14,64,64,64
# [1, 512, 8, 8] [1, 256, 16, 16] [1, 256, 32, 32]

class ASFF(nn.Module):
    def __init__(self, level, rfb=False,dim=[]):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = dim
        #[64, 32, 16]
        #[256,128,64]
        self.inter_dim = self.dim[self.level]
        
        if level==0:
            self.stride_level_1 = self.add_conv(self.dim[1], self.inter_dim, 3, 2, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 512
            self.stride_level_2 = self.add_conv(self.dim[2], self.inter_dim, 3, 2, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 128 -> 512
            self.expand = self.add_conv(self.inter_dim, self.dim[0]*2, 3, 1, leaky=False)              # nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 512 -> 1024
        elif level==1:
            self.compress_level_0 = self.add_conv(self.dim[0], self.inter_dim, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 512 -> 256
            self.stride_level_2 = self.add_conv(self.dim[2], self.inter_dim, 3, 2, leaky=False)    # nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 128 -> 256
            self.expand = self.add_conv(self.inter_dim, 64, 3, 1, leaky=False)                     # nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 512
        elif level==2:
            self.compress_level_0 = self.add_conv(self.dim[0], self.inter_dim, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 512 -> 128
            self.compress_level_1 = self.add_conv(self.dim[1], self.inter_dim, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 128
            self.expand = self.add_conv(self.inter_dim, self.dim[1], 3, 1,leaky=False)                     # nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 128 -> 256

        compress_c = 8 if rfb else 16                                                           #when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = self.add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 8
        self.weight_level_1 = self.add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 8
        self.weight_level_2 = self.add_conv(self.inter_dim, compress_c, 1, 1, leaky=False)# nn.Conv2d + nn.BatchNorm2d + nn.ReLU6, 256 -> 8

        self.weight_levels = nn.Conv2d(compress_c*3, 3, kernel_size=1, stride=1, padding=0)


    def add_conv(self,in_ch, out_ch, ksize, stride, leaky=True):
        stage = nn.Sequential()
        pad = (ksize - 1) // 2
        stage.add_module('conv', nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ksize, 
                                        stride=stride,padding=pad, bias=False))
        stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
        if leaky:
            stage.add_module('leaky', nn.LeakyReLU(0.1))
        else:
            stage.add_module('relu6', nn.ReLU6(inplace=True))
        return stage
        
    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level==0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized =F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v),1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+level_1_resized * levels_weight[:,1:2,:,:]+level_2_resized * levels_weight[:,2:,:,:]

        out = self.expand(fused_out_reduced)
        return out

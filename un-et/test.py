#--------------------------------------------#
#   该部分代码只用于看网络结构，并非测试代码
#--------------------------------------------#
from utils.dataloader import json_file
from nets.unet import Unet

if __name__ == "__main__":
    json_path=r"D:\torch_keras_code\KiTs19_seg\json_configs\kits_2d_unet.json"
    d=json_file(json_path)
    print(d)
    #model = Unet(num_classes=2).train().cuda()

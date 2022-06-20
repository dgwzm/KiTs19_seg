from models.networks import _get_model_instance
from models.networks.unet_CT_multi_att_dsv_3D import unet_CT_multi_att_dsv_3D
from models.networks.unet_CT_single_att_dsv_2D import unet_CT_2D_ASFF
from models.networks.unet_CT_2D_add_pic import unet_CT_2D_add_pic
from models.networks.sononet import sononet
from models.networks.unet_nonlocal_2D import unet_nonlocal_2D
import matplotlib.pyplot as plt
import nibabel as nib
#import pydicom as dic
import SimpleITK as sitk
import numpy as np
import os
import torch
import torchsample.transforms as ts

import cv2 as cv
import imageio
def tans_dcm_nii(path):
    reader=sitk.ImageFileReader()
    dic_name=reader.GetFileName(path)

    reader.SetFileName(dic_name)

    Image_array=np.array();
    out=sitk.GetImageFromArray(Image_array)


def read_d(path):

    nim=nib.load(path)
    images=nim.get_data()
    # meta = {'affine': nim.get_affine(),
    #         'dim': nim.header['dim'],
    #         'pixdim': nim.header['pixdim'],
    #         }
    # print(meta['affine'])
    # print(meta['dim'])
    # print(meta['pixdim'])
    #print(images.shape)
    return images


def Pi_s():
    data_path="/media/DATA/linda/wzm/Pancreas-CT/"

    label_path="/media/DATA/linda/wzm/TCIA_labels/"


def show_image(img):
    plt.imshow(img[:,:,120],cmap='gray')
    plt.show()

def read_dic(path):
    images=dic.read_file(path).pixel_array
    return images

def show_image_dic(img):
    plt.imshow(img,cmap='gray')
    plt.show()

def Dcm_to_nii(path_save="",path_read=""):
    ser_id=sitk.ImageSeriesReader.GetGDCMSeriesIDs(path_read)
    ser_file_name=sitk.ImageSeriesReader.GetGDCMSeriesFileNames(path_read,ser_id[0])
    ser_reader=sitk.ImageSeriesReader()
    ser_reader.SetFileNames(ser_file_name)
    image3D=ser_reader.Execute()
    sitk.WriteImage(image3D,path_save)

def Using_C():
    all_path="/media/DATA/linda/wzm/Pancreas-CT/"
    file_dir="/media/DATA/linda/wzm/Pancreas-CT/"

    save_path="/media/DATA/linda/wzm/Pancreas-CT-data/"

    s_path=[all_path+s for s in os.listdir(all_path)]
    ss_path=[s+'/'+os.listdir(s)[0]+'/'+os.listdir(s+'/'+os.listdir(s)[0])[0] for s in s_path if os.path.isdir(s)]
    save_p_list=[save_path+s+".nii" for s in os.listdir(all_path)]
    for file_path,s_path in zip(ss_path,save_p_list):
        Dcm_to_nii(path_save=s_path,path_read=file_path)

    print(ss_path)
    print(save_p_list)
    print(len(ss_path))
    for p in ss_path:
        if not os.path.isdir(p):
            print("Error",p)
    pass

class RandomCrop_2(object):

    def __init__(self, size):

        self.size = size

    def __call__(self, *inputs):
        #i=np.random.randint(0,256+1)
        #j=np.random.randint(0,256+1)

        h_idx = np.random.randint(0,self.size[0]+1)
        w_idx = np.random.randint(0,self.size[1]+1)
        outputs = []
        idx=0
        for idx, _input in enumerate(inputs):
            _input = _input[h_idx:(h_idx+self.size[0]),w_idx:(w_idx+self.size[1])]
            #_input = _input[:, h_idx:(h_idx + self.size[0]), w_idx:(w_idx + self.size[1])]
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]


def main(net_name="unet"):

    net_name="unet"
    png_name = "4_0375.png"
    png_name = "4_0398.png"   #0
    png_name = "5_0104.png"
    #png_name = "5_0375.png"
    #png_name = "3_0156.png"   # 0
    png_name = "3_0146.png"
    #png_name = "4_0375.png"
    #png_name = "4_0410.png"  # 111
    png_name = "4_0450.png"
    #png_name = "5_0202.png"
    #png_name = "8_0086.png"   #111
    #png_name = "11_0056.png"  # 111
    #png_name = "11_0108.png"  # 111
    #png_name = "12_0101.png"  # 111
    #png_name = "13_0269.png"  # 111
    U_models=unet_nonlocal_2D(n_classes=2,
                              is_batchnorm=True,
                              in_channels=1,
                              is_deconv=False,
                              nonlocal_mode="embedded_gaussian",
                              feature_scale=4)
    U_models=U_models.cuda()
    pth = "/media/DATA/linda/wzm/Attention-Gated-Networks/src/nonlocal_2D/nonlocal_2D/002_net_-0.50000.pth"
    U_models.load_state_dict(torch.load(pth))
    U_models.eval()


    #models.init_params()
    img=cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/data/"+png_name)[:,:,0]#512,512,3-> 512,512,  0,255

    img=torch.from_numpy(img).float() / 255.   # 512,512  ,0,1

    img=img.unsqueeze(0).unsqueeze(0).to("cuda:0") #1,1,512,512
    unet_img=U_models(img)

    img = U_models.apply_argmax_softmax(unet_img)

    _, u_predic = torch.max(img, dim=1)       #1,2,512,512 -> 1,1,512,512

    u_predic = u_predic.squeeze(0).cpu().detach().numpy().astype(np.uint8)
    u_predic[u_predic == 1] = 255  #1,512,512

    label_imgs = cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/label/"+png_name)[:, :, 0]
    img1=cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/data/"+png_name)[:,:,0]
    img2 = cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/data/" + png_name)[:, :, 0]
    cv.imshow("data1", img1)
    cv.imshow("data2", img2)
    cv.imshow("label", label_imgs)
    cv.imshow("unet", u_predic)

    #cv.imwrite("/media/DATA/linda/wzm/SAR-U-Net-segmentation/save_pth/unet.png",u_predic)
    #cv.imwrite("/media/DATA/linda/wzm/SAR-U-Net-segmentation/save_pth/sip.png", s_predic)

    cv.waitKey(0)

if __name__=='__main__':
    #model=unet_CT_att_dsv_2D(4,3,1)
    #model=unet_CT_2D_add_pic(feature_scale=4,n_classes=3,in_channels=1)
    model=unet_CT_2D_ASFF(feature_scale=4,n_classes=3,in_channels=1)
    x= torch.rand(4,1,512,512)
    #png="/home/linda/wzm/LITS/train_512/data/36_0072.png"
    #x=cv.imread(png)[:,:,0]
    
    y=model(x)
    print("y shape",y.shape)
    #main()

"""

if __name__=='__main__':
    x = torch.rand(1, 1,512, 512)
    print("dim",x.dim())
    model_name="sononet2"
    dim="2D"
    dcm_path="/media/DATA/linda/wzm/Pancreas-CT/PANCREAS_0064/11-24-2015-PANCREAS0064-Pancreas-67355/Pancreas-57695/1-015.dcm"
    #model=_get_model_instance(model_name,dim)

    #unet_models=unet_nonlocal_2D(4,2,True,1,True)
    #models=unet_CT_multi_att_dsv_3D()
    #print(unet_models)
    #y=unet_models(x)
    #print(y.shape)
    #path="/media/DATA/linda/wzm/TCIA_labels/label0002.nii.gz"
    #path = "/media/DATA/linda/wzm/TCIA_labels/label0001.nii"
    #data=read_d(path)
    #show_image(data)

    #Using_C()
    #data=read_dic(dcm_path)
    #show_image_dic(data)
    #img_t = cv.imread("/media/DATA/linda/wzm/LITS-17/train_512/data/31_0040.png")[:, :, 0]
    #img_v = cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/data/31_0040.png")[:, :, 0]

    f=os.path.exists("/media/DATA/linda/wzm/LITS-17/val_512/data/4_0375.png")
    print(f)

    f=os.path.exists("/media/DATA/linda/wzm/LITS-17/val_512/label/4_0375.png")
    print(f)

    img_t = cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/data/4_0375.png")  #[:, :, 0]
    img_v = cv.imread("/media/DATA/linda/wzm/LITS-17/val_512/label/4_0375.png") #[:, :, 0]
    #pth="/media/DATA/linda/wzm/Attention-Gated-Networks/step.py"

    #img_t = torch.from_numpy(img_t).float().unsqueeze(0).unsqueeze(0)
    #img_v = torch.from_numpy(img_v).float().unsqueeze(0).unsqueeze(0)
    #img_t = np.array(img_t)
    #img_v = np.array(img_v)
"""


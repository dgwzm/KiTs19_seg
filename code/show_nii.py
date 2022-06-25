import numpy as np
import os
import ntpath
import time
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
from nibabel.viewers import OrthoSlicer3D
from PIL import Image
import visdom


def show_pic(list):
    root = path= os.path.dirname(os.path.abspath(__file__))

    vis = visdom.Visdom(env='test2')
    num=0
    for i in list:
        
        print(num)
        epi_img_1 = nib.load(i[0])
        epi_img_data = epi_img_1.get_fdata()

        old=epi_img_data[150,:,:]
        img_old=np.array(old)
        max=np.max(img_old)
        min=np.min(img_old)
        up=img_old+1024
        down=2048
        new_img=(up/down)*255

        img_old[img_old<0]=0
        I=(img_old/1024)*255
        #img_pil = Image.fromarray(img_old)
        #I = img_pil.convert('L')
        img_L = np.array(I)
        #img_L=img_L[np.newaxis,:]
        
        epi_img_2 = nib.load(i[1])
        epi_img_data_2 = epi_img_2.get_fdata()
        d1=epi_img_data_2[150,:,:]
        #d1=d1[np.newaxis,:]
        d1[d1==1]=255
        d1[d1==2]=100
        data=np.concatenate((new_img,img_L),1)
        data=np.concatenate((data,d1),1)
        #data=torch.from_numpy(data)
        plt.figure(num)
        plt.imshow(data)
        num=num+1
        vis.image(data,opts={'title':"img %d"%(num)})

    plt.show()
    

def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")

def show_seg(list):
    for i in range(len(list)):
        epi_img_1 = nib.load(list[i])
        epi_img_data = epi_img_1.get_fdata()
        shapes=epi_img_data.shape
        old=epi_img_data[150,:,:]
        img_old=np.array(old)
        max=np.max(img_old)
        min=np.min(img_old)
        up=img_old+1024
        down=2048
        new=(up/down)*255
        img_pil = Image.fromarray(img_old)
        I = img_pil.convert('L')
        img_L = np.array(I)
        plt.figure(i)
        plt.subplot(1,2,1)
        plt.imshow(new)
        plt.subplot(1,2,2)
        plt.imshow(img_L)

        #plt.imshow(img,cmap='gray')
    plt.show()

if __name__=='__main__':
    img_list=[
        [r"/home/linda/wzm/kits19/data/master_00000.nii",
        r"/home/linda/wzm/kits19/label/segmentation_00000.nii"],
        [r"/home/linda/wzm/kits19/data/master_00003.nii",
        r"/home/linda/wzm/kits19/label/segmentation_00003.nii"], 
        [r"/home/linda/wzm/kits19/data/master_00008.nii",
        r"/home/linda/wzm/kits19/label/segmentation_00008.nii"],
        [r"/home/linda/wzm/kits19/data/master_00005.nii",
        r"/home/linda/wzm/kits19/label/segmentation_00005.nii"]]
    img_l=[
        r"D:\BaiduYunDownload\KiTs_old_data\case_00000\master_00000\imaging.nii",
        r"D:\BaiduYunDownload\KiTs_old_data\case_00000\segmentation.nii",
    ]


    #show_seg(img_l)
    show_pic(img_list)
    





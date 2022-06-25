from PIL import Image
import visdom
import numpy as np
import os
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import time

def show_seg(list):
    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000"
    for i in range(len(list)):
        Path=os.path.join(path,list[i])
        epi_img_1 = nib.load(Path)
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

def save_np(list):
    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000"
    for i in range(len(list)):
        Path=os.path.join(path,list[i])
        epi_img_1 = nib.load(Path)
        img_data = epi_img_1.get_fdata()
        if "segmentation" in list[i]:
            img=np.array(img_data,dtype=np.uint8)
            f=open(os.path.join(path,"segmen_%d.npy"%(i)),"wb")
            np.save(f,img)
            f.close()
        else:
            te_img_data=img_data[:]
            te_img_data[te_img_data<-255]=-255
            te_img_data[te_img_data>255]=255
            up=te_img_data+255
            down=510
            I=(up/down)*255
            way_1 = np.array(I,dtype=np.uint8)

            img_data[img_data<0]=0
            img_data[img_data>255]=255
            way_2 = np.array(img_data,dtype=np.uint8)

            f=open(os.path.join(path,"imaging_0.npy"),"wb")
            np.save(f,way_1)
            f.close()
            f=open(os.path.join(path,"imaging_1.npy"),"wb")
            np.save(f,way_2)
            f.close()

def test_np(list):
    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000"
    for i in range(len(list)):
        Path=os.path.join(path,list[i])
        imgs = np.load(Path)
        img=imgs[150,:,:]
        plt.figure(i)
        plt.imshow(img)
    plt.show()

def test_show_way():
    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000\imaging_0.npy"
    img = np.load(path)

    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000\imaging.nii"
    epi_img_1 = nib.load(path)
    img = epi_img_1.get_fdata()

    old=img[150,:,:]
    max=np.max(old)
    min=np.min(old)
    print("Max:",max,"Min:",min)

    test_old=old[:]
    test_old[test_old<-255]=-255
    test_old[test_old>255]=255
    up=test_old+255
    down=510
    I=(up/down)*255
    way_1 = np.array(I,dtype=np.uint8)

    old[old<0]=0
    old[old>255]=255
    way_2 = np.array(old,dtype=np.uint8)

    plt.figure(0)
    plt.subplot(1,2,1)
    plt.imshow(way_1)
    plt.subplot(1,2,2)
    plt.imshow(way_2)

    plt.show()

def img_nii_png(list=None):
    path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000\imaging.nii"
    way_0_path=r"D:\torch_keras_code\KiTs_old_data\case_00000\save_img_0"
    way_1_path=r"D:\torch_keras_code\KiTs_old_data\case_00000\save_img_1"
    epi_img = nib.load(path)
    img = epi_img.get_fdata()
    id_dex=0
    for old in img:
        id_dex=id_dex+1
        test_old=old[:]
        test_old[test_old<-255]=-255
        test_old[test_old>255]=255
        up=test_old+255
        down=510
        I=(up/down)*255
        way_0 = np.array(I,dtype=np.uint8)
        img_0 = Image.fromarray(way_0).convert('L')
        save_path=os.path.join(way_0_path,"%.3d.jpg"%(id_dex))
        img_0.save(save_path)

        old[old<0]=0
        old[old>255]=255
        way_1 = np.array(old,dtype=np.uint8)
        img_1 = Image.fromarray(way_1).convert('L')
        save_path=os.path.join(way_1_path,"%.3d.jpg"%(id_dex))
        img_1.save(save_path)


if __name__=='__main__':
    img_l=["imaging.nii","segmentation.nii"]
    np_list=["imaging_0.npy","imaging_1.npy","segmen_1.npy"]
    img_path=r"D:\torch_keras_code\KiTs_old_data\case_00000\save_img_0\001.jpg"
    start = time.time()
    imgs=cv2.imread(img_path,flags=0)
    print(imgs.shape)
    end = time.time()
    print("npy use time:",end-start)

    start = time.time()
    image = Image.open(img_path).convert('L')
    print(imgs.shape)
    end = time.time()
    print("npy use time:",end-start)
    #img_nii_png()
    #test_show_way()
    #save_np(img_l)
    #test_np(np_list)
    #show_seg(img_l)

"""
    start = time.time()
    np_path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000\imaging_0.npy"
    img = np.load(np_path)
    end = time.time()
    print("npy use time:",end-start)

    start = time.time()
    nii_path=r"D:\torch_keras_code\KiTs_old_data\case_00000\master_00000\imaging.nii"
    epi_img = nib.load(nii_path)
    imgs = epi_img.get_fdata()
    end = time.time()
    print("nii use time:",end-start)
"""

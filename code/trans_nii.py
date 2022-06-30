import numpy as np
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm

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

def save_img_jpg():

    o_img_path="/home/linda/wzm/kits19/data"
    jpg_path_0="/home/linda/wzm/kits19/data_jpg_0"
    jpg_path_1="/home/linda/wzm/kits19/data_jpg_1"
    file_name_list=os.listdir(o_img_path)
    for f in tqdm(file_name_list):
        if "master" in f:
            file_nam=f[:-3]
            new_dir_0=os.path.join(jpg_path_0,file_nam)
            new_dir_1=os.path.join(jpg_path_1,file_nam)
            if os.path.exists(new_dir_0) and os.path.exists(new_dir_1):
                continue
            else:
                os.makedirs(new_dir_0)
                os.makedirs(new_dir_1)
            nii_file=os.path.join(o_img_path,f)
            epi_img = nib.load(nii_file)
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
                save_path=os.path.join(new_dir_0,"%.3d.png"%(id_dex))
                img_0.save(save_path)

                old[old<0]=0
                old[old>255]=255
                way_1 = np.array(old,dtype=np.uint8)
                img_1 = Image.fromarray(way_1).convert('L')
                save_path=os.path.join(new_dir_1,"%.3d.png"%(id_dex))
                img_1.save(save_path)

def save_label_jpg():
    o_img_path="/home/linda/wzm/kits19/label"
    jpg_path="/home/linda/wzm/kits19/label_jpg"
    file_name_list=os.listdir(o_img_path)
    for f in tqdm(file_name_list):
        if "segment" in f:
            file_nam=f[:-3]
            new_dir=os.path.join(jpg_path,file_nam)
            if os.path.exists(new_dir):
                continue
            else:
                os.makedirs(new_dir)
            nii_file=os.path.join(o_img_path,f)
            epi_img = nib.load(nii_file)
            imgs = epi_img.get_fdata()
            id_dex=0
            for img in imgs:
                id_dex=id_dex+1
                img[img==1]=125
                img[img==2]=255
                way_0 = np.array(img,dtype=np.uint8)
                save_path=os.path.join(new_dir,"%.3d.png"%(id_dex))
                cv2.imwrite(save_path,way_0)
                #img_0.save(save_path)


if __name__=='__main__':
    print("start load data")
    save_img_jpg()
    print("start load label")
    save_label_jpg()




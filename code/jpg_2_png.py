import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os
from tqdm import tqdm

def change_png(file_p):
    i=cv2.imread(file_p)
    i=i[:,:,0]
    d=np.array(i)
    new_img=np.zeros(d.shape)
    for i in range(512):
        for j in range(512):
            if d[i,j] <100:
                new_img[i,j]=0
            elif d[i,j] >=100 and d[i,j] <200:
                new_img[i,j]=125
            elif d[i,j] >=200 :
                new_img[i,j]=255
    return new_img


def save_label_png(jpg_path):
    file_name_list=os.listdir(jpg_path)
    for f in file_name_list:
        new_dir=os.path.join(jpg_path,f)
        jpg_list=os.listdir(new_dir)
        for i in tqdm(jpg_list):
            file_name=i.split('.')[0]
            file_n=os.path.join(jpg_path,f,i)
            new_imgs=change_png(file_n)
            os.remove(file_n)
            cv2.imwrite(os.path.join(jpg_path,f,file_name+".png"),new_imgs)

path=r"D:\torch_keras_code\kits_data_label\label_jpg"
save_label_png(path)

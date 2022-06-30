import torch
import torch.nn as nn
import numpy as np
import cv2
import os
import random
import cv2 as cv
from os import listdir
from os.path import join

class LITS_dataset(torch.utils.data.Dataset):
    #root_dir, split, transform=None, preload_data=False
    def __init__(self, root_dir, split,transform=None,preload_data=False,wh=256):
        self.wh = wh
        self.path_raw = join(root_dir, split, 'data')
        self.path_label = join(root_dir, split, 'label')

        self.image_filenames  = sorted([join(self.path_raw, x) for x in listdir(self.path_raw)])
        self.target_filenames = sorted([join(self.path_label, x) for x in listdir(self.path_raw)])

        #self.image_filenames=self.image_filenames[:200]
        #self.target_filenames=self.target_filenames[:200]
        assert len(self.image_filenames) == len(self.target_filenames)

        #self.path_raw = path_raw
        #self.path_label = path_label
        #self.filelist = sorted(x for x in listdir(self.path_label))

        self.transform=transform

        #print(self.filelist[:5])

    def __len__(self):
        return len(self.image_filenames)  #len(self.filelist)

    def read_data(self, index):
        #i=np.random.randint(0,256+1)
        #j=np.random.randint(0,256+1)
        i=np.random.randint(0,512-self.wh+1)
        j=np.random.randint(0,512-self.wh+1)
        #raw = cv2.imread(os.path.join(self.path_raw, self.filelist[index]))     #[:, :, 0]  # 直接返回numpy.ndarray 对象
        raw = cv2.imread(self.image_filenames[index])
        raw = raw[j:j + self.wh,i:i + self.wh]
        #raw  = raw.astype(np.float16)
        raw = torch.from_numpy(raw).float() / 255.

        #label = cv2.imread(os.path.join(self.path_label, self.filelist[index]))  #[:, :, 0]

        label = cv2.imread(self.target_filenames[index])
        label = label[j:j + self.wh, i:i + self.wh]
        label = torch.from_numpy(label).float()
        # print('read_raw: ' + str(raw.shape) + '|' + 'label' + str(label.shape))

        return raw, label

    def __getitem__(self, index):
        raw, label = self.read_data(index)
        raw = raw.unsqueeze(0)

        #label[label > 0] = 1
        #raw.half()
        #label.half()
        # label = label.unsqueeze(0)
        # print(raw.shape, label.shape) torch.Size([4,1,256,256]),torch.Size([4,256,256])

        return raw, label


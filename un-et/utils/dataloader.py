import os
import random
from random import shuffle
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from PIL import Image
#from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
import json
import collections
from os.path import join
from os import listdir
import torch
import torchvision.transforms as transform

def json_file(filename):
    def _json_object_hook(d):
        return collections.namedtuple('X', d.keys())(*d.values())
    def json2obj(data):
        return json.loads(data, object_hook=_json_object_hook)
    return json2obj(open(filename).read())

def json_dict(file_name):
    js_f=open(file_name,"r")
    ds=json.load(js_f,object_hook=lambda x:collections.namedtuple('d', x.keys())(*x.values()))
    return ds

def letterbox_image(image, label , size):
    label = Image.fromarray(np.array(label))
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    label = label.resize((nw,nh), Image.NEAREST)
    new_label = Image.new('L', size, (0))
    new_label.paste(label, ((w-nw)//2, (h-nh)//2))

    return new_image, new_label

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class DeeplabDataset(Dataset):
    def __init__(self,train_lines,image_size,num_classes,random_data,dataset_path):
        super(DeeplabDataset, self).__init__()

        self.train_lines    = train_lines
        self.train_batches  = len(train_lines)
        self.image_size     = image_size
        self.num_classes    = num_classes
        self.random_data    = random_data
        self.dataset_path   = dataset_path

    def __len__(self):
        return self.train_batches

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        label = Image.fromarray(np.array(label))

        h, w = input_shape
        # resize image
        rand_jit1 = rand(1-jitter,1+jitter)
        rand_jit2 = rand(1-jitter,1+jitter)
        new_ar = w/h * rand_jit1/rand_jit2

        scale = rand(0.5,1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        label = label.resize((nw,nh), Image.NEAREST)
        label = label.convert("L")
        
        # flip image or not
        flip = rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # place image
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_label = Image.new('L', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # distort image
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255
        return image_data,label

    def __getitem__(self, index):
        if index == 0:
            shuffle(self.train_lines)
            
        annotation_line = self.train_lines[index]
        name = annotation_line.split()[0]

        # 从文件中读取图像
        jpg = Image.open(os.path.join(os.path.join(self.dataset_path, "JPEGImages"), name + ".jpg"))
        png = Image.open(os.path.join(os.path.join(self.dataset_path, "SegmentationClass"), name + ".png"))

        if self.random_data:
            jpg, png = self.get_random_data(jpg,png,(int(self.image_size[1]),int(self.image_size[0])))
        else:
            jpg, png = letterbox_image(jpg, png, (int(self.image_size[1]),int(self.image_size[0])))

        png = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels = np.eye(self.num_classes+1)[png.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.image_size[1]),int(self.image_size[0]),self.num_classes+1))

        jpg = np.transpose(np.array(jpg),[2,0,1])/255

        return jpg, png, seg_labels

class Kits_2d_dataset(Dataset):
    def __init__(self, types="train",opts=None):
        super(Kits_2d_dataset, self).__init__()
        self.opt=opts
        self.wh = opts.data_set_trans.imge_size
        self.random_wh = opts.data_set_trans.random_img_size

        self.data_txt_path=join(opts.train.txt_dir, opts.train.data_txt_name)
        self.label_txt_path=join(opts.train.txt_dir, opts.train.label_txt_name)

        self.data_dir   = join(opts.train.data_dir, opts.train.data_path_name)
        self.label_dir = join(opts.train.data_dir, opts.train.label_path_name)

        self.d_name_list=open(self.data_txt_path,"r").readlines()
        self.l_name_list=open(self.label_txt_path,"r").readlines()
        print("d_name_list len:",len(self.d_name_list))
        print("l_name_list len:",len(self.l_name_list))

        data_len=len(self.l_name_list)
        train_len=int(data_len*opts.train.train_val)
        val_len=data_len-train_len

        if types=="train":
            self.image_filenames  = self.d_name_list[:train_len]
            self.target_filenames = self.l_name_list[:train_len]

        else:
            self.image_filenames  = self.d_name_list[train_len:train_len+val_len]
            self.target_filenames = self.l_name_list[train_len:train_len+val_len]

        assert len(self.image_filenames) == len(self.target_filenames)

        #print(self.filelist[:5])

    def __len__(self):
        return len(self.image_filenames)  #len(self.filelist)

    def read_data(self, index):

        i=np.random.randint(0,self.wh-self.random_wh+1)
        j=np.random.randint(0,self.wh-self.random_wh+1)

        data_jpg_path =os.path.join(self.data_dir,self.image_filenames[index][:-1])
        label_jpg_path=os.path.join(self.label_dir,self.target_filenames[index][:-1])

        raw = Image.open(data_jpg_path)
        print("data:",raw.size)
        raw=np.array(raw)
        raw = raw[j:j + self.random_wh,i:i + self.random_wh]
        raw = torch.from_numpy(raw).float() / 255.

        label = Image.open(label_jpg_path)
        print("label:",label.size)
        label=np.array(label)
        label[label<100]=0
        label[label>100 and label<200]=125
        label[label>200]=255
        label = label[j:j + self.random_wh, i:i + self.random_wh]

        for x in label:
            for i in x:
                if i not in [0,125,255]:
                    print("error d:",i)
                    raise "---err!---"
        label = torch.from_numpy(label).float()

        if self.opt.data_set_trans.random_trans:
            if random.uniform(0,1)<self.opt.data_set_trans.RandomHorizontalFlip:
                raw=transform.RandomHorizontalFlip(1)(raw)
                label=transform.RandomHorizontalFlip(1)(label)

            if random.uniform(0,1)<self.opt.data_set_trans.RandomVerticalFlip:
                raw=transform.RandomVerticalFlip(1)(raw)
                label=transform.RandomVerticalFlip(1)(label)

            if random.uniform(0,1)<self.opt.data_set_trans.RandomRotation:
                raw=transform.RandomRotation(1)(raw)
                label=transform.RandomRotation(1)(label)

        return raw, label

    def __getitem__(self, index):
        raw, label = self.read_data(index)
        raw = raw.unsqueeze(0)
        return raw, label

class Kits_2d_two_seg_dataset(Dataset):
    #root_dir, split, transform=None, preload_data=False
    def __init__(self, root_dir, split,transform=None,preload_data=False,wh=256):
        super(Kits_2d_two_seg_dataset, self).__init__()
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

class Kits_3d_dataset(Dataset):
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

class LossHistory():
    def __init__(self, log_dir):
        import datetime
        curr_time = datetime.datetime.now()
        time_str = datetime.datetime.strftime(curr_time,'%Y_%m_%d_%H_%M_%S')
        self.log_dir    = log_dir
        self.time_str   = time_str
        self.save_path  = os.path.join(self.log_dir, "loss_" + str(self.time_str))
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.save_path)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        with open(os.path.join(self.save_path, "epoch_val_loss_" + str(self.time_str) + ".txt"), 'a') as f:
            f.write(str(val_loss))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.save_path, "epoch_loss_" + str(self.time_str) + ".png"))

# DataLoader中collate_fn使用
def deeplab_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels

def get_dataset(name):
    return {
        'Deeplab': DeeplabDataset,
        'Kits_2d': Kits_2d_dataset,
        'Kits_2d_two': Kits_2d_two_seg_dataset,
        'Kits_3d': Kits_3d_dataset,
    }[name]


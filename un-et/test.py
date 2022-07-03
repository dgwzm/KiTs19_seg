
import os
import time
import numpy as np
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from utils.dataloader import deeplab_dataset_collate,json_dict,get_dataset
from utils.metrics import *
from nets.get_net import get_network

class Predict:
    def __init__(self, opts):
        self.opt=opts
        self.model          = get_network(opts)
        self.pre_loss_list=[]
        self.pre_acc_list =[]
        self.val_pre_path=r"../../val_pre_label/"
        self.test_pre_path=r"../../test_pre_label/"

        self.device = "cpu"
        if opts.train.use_gpu:
            device = 'cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu'
            self.device = torch.device(device)
            print('Compute device: ' + str(self.device))

        self.optimizer    = get_optimizer(self.model.parameters(),opts)
        self.lr_scheduler = get_scheduler(self.optimizer,opts)
        self.criterion    = get_criterion(opts,self.device)
        self.accuracy     = f_score
        self.data_class   = get_dataset(opts.train.dataset_name)

        if opts.model.continue_train:
            model_dict = self.model.state_dict()
            model_path = opts.model.path_pre_model
            print("path:",model_path)

            pretrained_dict = torch.load(model_path,map_location=self.device)["model_dict"]
            pre_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items()}
            self.model.load_state_dict(pre_dict)

        if opts.train.use_gpu:
            self.model = self.model.to(self.device)

        self.train_data_calss   = self.data_class(opts=opts,way="train")
        self.val_data_calss     = self.data_class(opts=opts,way="pre_val")
        self.test_data_calss    = self.data_class(opts=opts,way="pre_test")

        self.type_name        = "%s-%s-%s-%s"%(self.opt.model.model_type,self.opt.train.criterion,self.opt.train.lr_policy,self.opt.train.optim)
        self.data_type_name   ="%s-%s-%s-%s-%s"%(self.opt.train.dataset_name,self.opt.model.model_type,self.opt.train.criterion,self.opt.train.lr_policy,self.opt.train.optim)
        print("batch_size:",opts.train.batchSize)
        print("dataset name:",self.opt.train.dataset_name)
        print("type name:",self.type_name)

    def start_pre(self):
        self.model.eval()
        t_loss=0
        #tbar_val = tqdm(self.val_data_calss)

        # for iteration, (seg_data, label, f_name) in enumerate(tbar_val):
        #     seg_data =Variable(seg_data,requires_grad=True).to(self.device)
        #     label    =Variable(label,requires_grad=True).to(self.device)
        #     if seg_data.shape[2]!=512:
        #         print(f_name)
        #         raise "stop"
        #     prediction = self.model(seg_data)
        #     loss = self.criterion(prediction, label)
        #     t_loss=t_loss+loss.item()
        #     _, prediction = torch.max(prediction, dim=1)
        #     prediction = prediction.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)
        #     prediction[prediction == 1] = 125
        #     prediction[prediction == 2] = 255  #1,512,512
        #     f_id=f_name.split("/")
        #     f_dir=os.path.join(self.val_pre_path,f_id[0])
        #     if not os.path.exists(f_dir):
        #         os.makedirs(f_dir)
        #     f_name=f_name.split(".")[0]
        #     save_path=os.path.join(self.val_pre_path,"%s.png"%(f_name))
        #
        #     #print("shape:",prediction.shape)
        #     #print("path:",save_path)
        #     #raise "Stop"
        #     cv2.imwrite(save_path,prediction)
        #
        # print("Pre:",t_loss/len(self.val_data_calss))

        tbar_test = tqdm(self.test_data_calss)

        for iteration, (seg_data, f_name) in enumerate(tbar_test):
            seg_data =Variable(seg_data,requires_grad=True).to(self.device)
            prediction = self.model(seg_data)
            _, prediction = torch.max(prediction, dim=1)
            prediction = prediction.squeeze(0).squeeze(0).cpu().detach().numpy().astype(np.uint8)

            prediction[prediction == 1] = 125
            prediction[prediction == 2] = 255  #1,512,512

            f_id=f_name.split("/")
            f_dir=os.path.join(self.test_pre_path,f_id[0])
            if not os.path.exists(f_dir):
                os.makedirs(f_dir)
            f_name=f_name.split(".")[0]
            save_path=os.path.join(self.test_pre_path,"%s.png"%(f_name))
            cv2.imwrite(save_path,prediction)

if __name__ == "__main__":
    json_path=r"../json_configs/kits_2d_unet.json"
    opts=json_dict(json_path)
    pre = Predict(opts)
    pre.start_pre()
    #model = Unet(num_classes=2).train().cuda()

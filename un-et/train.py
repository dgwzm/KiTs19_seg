import json
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from nets.unet_training import CE_Loss, Dice_loss, LossHistory
from utils.dataloader import deeplab_dataset_collate,json_file,json_dict,get_dataset
from utils.metrics import *
from nets.get_net import get_network
import collections
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, opts):
        self.opt=opts
        self.model        = get_network(opts)
        self.train_loss_list=[]
        self.val_loss_list  =[]
        self.train_acc_list =[]
        self.val_acc_list   =[]
        self.lr_list        =[]
        self.best_loss_list =[]
        self.best_loss=10

        self.device = "cpu"
        if opts.train.use_gpu:
            device = 'cuda:{}'.format(opts.train.gpu_id) if torch.cuda.is_available() else 'cpu'
            print('Compute device: ' + device)
            self.device = torch.device(self.device)
            self.model = torch.nn.DataParallel(self.model)
            #cudnn.benchmark = True
            self.model = self.model.to(self.device)

        if opts.model.continue_train:
            model_dict = self.model.state_dict()
            model_path = opts.model.path_pre_model
            print("path:",model_path)
            pretrained_dict = torch.load(model_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        self.optimizer    = get_optimizer(self.model.parameters(),opts)
        self.lr_scheduler = get_scheduler(self.optimizer,opts)
        self.criterion    = get_criterion(opts)
        self.accuracy     = f_score
        self.data_class   = get_dataset(opts.train.dataset_name)

        self.train_data   = self.data_class("train",opts)
        self.val_data     = self.data_class("val",opts)

        self.train_dataset= DataLoader(self.train_data, batch_size=opts.train.batchSize, num_workers=opts.train.num_workers,
                                       shuffle=True)

        self.val_dataset  = DataLoader(self.val_data, batch_size=opts.train.batchSize, num_workers=opts.train.num_workers,
                                       shuffle=False)

        self.epoch_size_train = len(self.train_dataset) // opts.train.batchSize
        self.epoch_size_val   = len(self.val_dataset)   // opts.train.batchSize
        self.n_epochs         = opts.train.n_epochs
        print("epoch_size_train:",self.epoch_size_train,"epoch_size_val:",self.epoch_size_val)

    def training(self, epoch):
        train_loss = 0.0
        train_score=0
        self.model.train()
        start_time = time.time()
        with tqdm(total=self.epoch_size_train,desc=f'Epoch {epoch + 1}/{self.n_epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, (seg_data, label) in enumerate(self.train_dataset):
                seg_data =Variable(seg_data,requires_grad=True).to(self.device)
                label    =Variable(label,requires_grad=True).to(self.device)
                self.optimizer.zero_grad()
                prediction = self.model(seg_data)
                loss = self.criterion(prediction, label)
                loss.backward()
                score= self.accuracy(prediction, label)
                self.optimizer.step()

                train_loss += loss.item()
                train_score+= score

                waste_time = time.time() - start_time
                op_lr=get_lr(self.optimizer)

                pbar.set_postfix(**{'val_loss' : train_loss / (iteration + 1),
                                    'val_score' : train_score / (iteration + 1),
                                    's/step'    : waste_time,
                                    'lr'        : op_lr})
                pbar.update(1)
                start_time = time.time()
            self.train_loss_list.append(train_loss / len(self.train_dataset))
            self.train_acc_list.append(train_score / len(self.train_dataset))
            self.lr_list.append(get_lr(self.optimizer))
            self.lr_scheduler.step()

    def validation(self, epoch):
        val_loss=0
        val_score=0
        self.model.eval()
        with tqdm(total=self.epoch_size_val, desc=f'Epoch {epoch + 1}/{self.n_epochs}',postfix=dict,mininterval=0.3) as pbar:
            for iteration, (seg_data, label) in enumerate(self.val_dataset):
                with torch.no_grad():
                    seg_data =Variable(seg_data,requires_grad=True).to(self.device)
                    label    =Variable(label,requires_grad=True).to(self.device)
                    prediction= self.model(seg_data)
                    loss      = self.criterion(prediction, label)
                    score     = self.accuracy(prediction, label)
                    val_loss += loss.item()
                    val_score+=score
                pbar.set_postfix(**{'val_loss' : val_loss / (iteration + 1),
                                    'val_score': val_score / (iteration + 1)})
                pbar.update(1)
            t_loss=val_loss / len(self.val_dataset)
            if t_loss<self.best_loss:
                self.best_loss=t_loss
                self.best_loss_list.append(self.best_loss)
                if len(self.best_loss_list)>1:
                    last_pth=os.path.join(self.opt.save_dir.save_pth_dir,"val_loss_%.4f.pth"%(self.best_loss_list[-2]))
                    if os.path.exists(last_pth):
                        os.remove(last_pth)
                    torch.save({'model_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict(),
                                'best_loss': self.best_loss,
                                'lr_d':self.lr_list[-1]},
                               os.path.join(self.opt.save_dir.save_pth_dir,"val_loss_%.4f.pth"%(self.best_loss)))

            self.val_loss_list.append(t_loss)
            self.val_acc_list.append(val_score / len(self.val_dataset))

    def save_plt(self):
        plt.figure(0)
        plt.plot(self.train_loss_list, 'b--', label="Train loss")
        plt.plot(self.val_loss_list, 'r--', label="Validation loss")
        plt.title('Learning loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"loss.jpg"))

        plt.figure(1)
        plt.plot(self.train_acc_list, 'b--', label="Train acc")
        plt.plot(self.val_acc_list, 'r--', label="Validation acc")
        plt.title('Learning acc')
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"acc.jpg"))

        plt.figure(2)
        plt.plot(self.lr_list, 'b--', label="Lr")
        plt.title('Learning Lr')
        plt.xlabel("Epoch")
        plt.ylabel("Lr")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"lr.jpg"))
        plt.show()


if __name__ == "__main__":
    json_filename=r"D:\torch_keras_code\KiTs19_seg\json_configs\kits_2d_unet.json"
    opts=json_dict(json_filename)
    trainer = Trainer(opts)
    tx=0
    da,la=trainer.train_data[322]
    print(da.shape)
    print(la.shape)
    da=da.squeeze(0)
    d=da.numpy()
    d=d*255
    d=d.astype(np.uint8)
    la=la.numpy().astype(np.uint8)
    plt.figure(0)
    plt.imshow(d)
    plt.figure(1)
    plt.imshow(la)
    plt.show()
    #     trainer.validation(epoch)
    # trainer.save_plt()


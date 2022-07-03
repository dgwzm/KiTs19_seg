import json
import os
import time
import numpy as np
from torch.utils.data import DataLoader,Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from nets.unet_training import CE_Loss, Dice_loss, LossHistory
from utils.dataloader import deeplab_dataset_collate,json_dict,get_dataset
from utils.metrics import *
from nets.get_net import get_network
import datetime
import visdom
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer:
    def __init__(self, opts):
        self.opt=opts
        self.model          = get_network(opts)
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
            self.device = torch.device(device)
            print('Compute device: ' + str(self.device))

        self.optimizer    = get_optimizer(self.model.parameters(),opts)
        self.lr_scheduler = get_scheduler(self.optimizer,opts)
        self.criterion    = get_criterion(opts,self.device)
        self.accuracy     = f_score
        self.data_class   = get_dataset(opts.train.dataset_name)

        if opts.model.continue_train:
            model_path = opts.model.path_pre_model
            print("path:",model_path)
            loss_d=model_path.split("_")[-1][:-4]
            self.best_loss=float(loss_d)
            self.best_loss_list.append(float(loss_d))

            pretrained_dict = torch.load(model_path,map_location=self.device)["model_dict"]
            pre_dict = {k.replace('module.',''): v for k, v in pretrained_dict.items()}
            self.model.load_state_dict(pre_dict)

        if opts.train.use_gpu:
            if opts.train.use_gpu_list and torch.cuda.device_count()>1:
                self.model = torch.nn.DataParallel(self.model,device_ids=[1,2,3])
            self.model = self.model.to(self.device)

        self.train_data_calss   = self.data_class(opts=opts,way="train")
        self.val_data_calss     = self.data_class(opts=opts,way="val")

        self.train_dataset= DataLoader(self.train_data_calss, batch_size=opts.train.batchSize,
                                       num_workers=opts.train.num_workers,shuffle=False)

        self.val_dataset  = DataLoader(self.val_data_calss, batch_size=opts.train.batchSize,
                                       num_workers=opts.train.num_workers,shuffle=False)

        self.epoch_size_train = len(self.train_dataset) // opts.train.batchSize
        self.epoch_size_val   = len(self.val_dataset)   // opts.train.batchSize
        self.n_epochs         = opts.train.n_epochs
        self.type_name        = "%s-%s-%s-%s"%(self.opt.model.model_type,self.opt.train.criterion,self.opt.train.lr_policy,self.opt.train.optim)
        self.data_type_name   ="%s-%s-%s-%s-%s"%(self.opt.train.dataset_name,self.opt.model.model_type,self.opt.train.criterion,self.opt.train.lr_policy,self.opt.train.optim)
        print("batch_size:",opts.train.batchSize)
        print("dataset name:",self.opt.train.dataset_name)
        print("type name:",self.type_name)

    def training(self, epoch):
        train_loss = 0.0
        train_score=0
        self.model.train()
        tbar = tqdm(self.train_dataset,desc=f'Epoch {epoch + 1}/{self.n_epochs}')
        for iteration, (seg_data, label) in enumerate(tbar):
            seg_data =Variable(seg_data,requires_grad=True).to(self.device)
            label    =Variable(label,requires_grad=True).to(self.device)
            self.optimizer.zero_grad()
            prediction = self.model(seg_data)
            loss = self.criterion(prediction, label)
            loss.backward()
            #score= self.accuracy(prediction, label)
            self.optimizer.step()
            score=2-loss.item()
            train_loss += loss.item()
            train_score+= score

            op_lr=get_lr(self.optimizer)
            tbar.set_description(
                    'train epoch:%d/%d t_loss:%.5f acc:%.5f lr:%.4f' %
                (epoch+1,self.n_epochs,train_loss / (iteration + 1),train_score / (iteration + 1),op_lr))

        a_t_loss=train_loss / len(self.train_dataset)
        print("train loss:",a_t_loss)
        self.train_loss_list.append(a_t_loss)
        self.train_acc_list.append(train_score / len(self.train_dataset))
        self.lr_list.append(get_lr(self.optimizer))
        self.lr_scheduler.step()

    def validation(self, epoch):
        val_loss=0
        val_score=0
        self.model.eval()

        tbar = tqdm(self.val_dataset,desc=f'Epoch {epoch + 1}/{self.n_epochs}')
        for iteration, (seg_data, label) in enumerate(tbar):
            with torch.no_grad():
                seg_data =Variable(seg_data,requires_grad=True).to(self.device)
                label    =Variable(label,requires_grad=True).to(self.device)
                prediction= self.model(seg_data)
                loss      = self.criterion(prediction, label)
                score=2-loss.item()
                val_loss += loss.item()
                val_score+=score

            tbar.set_description(
                    'val epoch:%d/%d v_loss:%.5f acc:%.5f' %
                (epoch+1,self.n_epochs,val_loss / (iteration + 1),val_score / (iteration + 1)))

        t_loss=val_loss / len(self.val_dataset)
        print("val loss:",t_loss)
        if t_loss<self.best_loss:
            self.best_loss=t_loss
            self.best_loss_list.append(self.best_loss)
            if len(self.best_loss_list)>1:
                last_pth=os.path.join(self.opt.train.dataset_name,self.opt.save_dir.save_pth_dir,
                                      "%s_%.4f.pth"%(self.type_name,self.best_loss_list[-2]))
                print("Last:",last_pth)
                print("New:",os.path.join(self.opt.train.dataset_name,self.opt.save_dir.save_pth_dir,
                                          "%s_%.4f.pth"%(self.type_name,self.best_loss)))

                if os.path.exists(last_pth):
                    os.remove(last_pth)
            torch.save({'model_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'best_loss': self.best_loss,
                        'lr_d':self.lr_list[-1]},
                         os.path.join(self.opt.train.dataset_name,self.opt.save_dir.save_pth_dir,
                                      "%s_%.4f.pth"%(self.type_name,self.best_loss)))

        self.val_loss_list.append(t_loss)
        self.val_acc_list.append(val_score / len(self.val_dataset))

    def save_plt(self):
        plt.figure(0)
        plt.plot(self.train_loss_list, 'b', label="Train loss")
        plt.plot(self.val_loss_list,   'r', label="Validation loss")
        plt.legend(["train loss","val loss"])
        plt.title('Train and Val loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"%s_loss.jpg"%(self.type_name)))

        plt.figure(1)
        plt.plot(self.train_acc_list, 'b', label="Train acc")
        plt.plot(self.val_acc_list, 'r', label="Validation acc")
        plt.legend(["train acc","val acc"])
        plt.title('Learning acc')
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"%s_acc.jpg"%(self.type_name)))

        plt.figure(2)
        plt.plot(self.lr_list, 'b', label="Lr")
        plt.title('Learning Lr')
        plt.xlabel("Epoch")
        plt.ylabel("Lr")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"%s_lr.jpg"%(self.type_name)))
        self.save_logs()


    def save_logs(self):

        now = datetime.datetime.now()
        StyleTime = now.strftime("%Y-%m-%d_%H:%M:%S")

        lr_csv=os.path.join(self.opt.train.txt_dir,self.opt.train.lr_csv)
        loss_csv=os.path.join(self.opt.train.txt_dir,self.opt.train.t_v_loss)
        acc_csv=os.path.join(self.opt.train.txt_dir,self.opt.train.t_v_acc)

        f_lr_csv  =open(lr_csv,"a+")
        f_loss_csv=open(loss_csv,"a+")
        f_acc_csv =open(acc_csv,"a+")

        f_lr_csv.writelines("%s,%s"%(StyleTime,self.data_type_name))
        for i in self.lr_list:
            f_lr_csv.writelines(",%f"%(i))
        f_lr_csv.writelines("\n")

        f_loss_csv.writelines("%s,%s,train"%(StyleTime,self.data_type_name))
        for i in self.train_loss_list:
            f_loss_csv.writelines(",%.8f"%(i))
        f_loss_csv.writelines("\n")

        f_loss_csv.writelines("%s,%s,val"%(StyleTime,self.data_type_name))
        for i in self.val_loss_list:
            f_loss_csv.writelines(",%.8f"%(i))
        f_loss_csv.writelines("\n")

        f_acc_csv.writelines("%s,%s,train"%(StyleTime,self.data_type_name))
        for i in self.train_acc_list:
            f_acc_csv.writelines(",%.8f"%(i))
        f_acc_csv.writelines("\n")

        f_acc_csv.writelines("%s,%s,val"%(StyleTime,self.data_type_name))
        for i in self.val_acc_list:
            f_acc_csv.writelines(",%.8f"%(i))
        f_acc_csv.writelines("\n")
        #plt.show()

if __name__ == "__main__":
    json_filename=r"../json_configs/kits_2d_unet.json"
    opts=json_dict(json_filename)
    trainer = Trainer(opts)
    for i in range(opts.train.n_epochs):
        trainer.training(i)
        trainer.validation(i)
    trainer.save_plt()


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
            if opts.train.use_gpu_list:
                self.model = torch.nn.DataParallel(self.model, device_ids=opts.train.gpu_list)
            #cudnn.benchmark = True
            self.model = self.model.cuda()

        self.optimizer    = get_optimizer(self.model.parameters(),opts)
        self.lr_scheduler = get_scheduler(self.optimizer,opts)
        self.criterion    = get_criterion(opts)
        self.accuracy     = f_score
        self.data_class   = get_dataset(opts.train.dataset_name)

        if opts.model.continue_train:
            model_dict = self.model.state_dict()
            model_path = opts.model.path_pre_model
            print("path:",model_path)
            loss_d=model_path.split("_")[2][:-4]
            self.best_loss=float(loss_d)
            self.best_loss_list.append(float(loss_d))

            pretrained_dict = torch.load(model_path, map_location=self.device)
            pretrained_dict = {k: v for k, v in pretrained_dict["model_dict"].items() if np.shape(model_dict[k]) ==  np.shape(v)}
            model_dict.update(pretrained_dict)
            self.model.load_state_dict(model_dict)

        self.train_data_calss   = self.data_class(opts=opts,way="train")
        self.val_data_calss     = self.data_class(opts=opts,way="val")

        self.train_dataset= DataLoader(self.train_data_calss, batch_size=opts.train.batchSize,
                                       num_workers=opts.train.num_workers,shuffle=False)

        self.val_dataset  = DataLoader(self.val_data_calss, batch_size=opts.train.batchSize,
                                       num_workers=opts.train.num_workers,shuffle=False)

        self.epoch_size_train = len(self.train_dataset) // opts.train.batchSize
        self.epoch_size_val   = len(self.val_dataset)   // opts.train.batchSize
        self.n_epochs         = opts.train.n_epochs
        print("epoch_size_train:",self.epoch_size_train,"epoch_size_val:",self.epoch_size_val)

    def training(self, epoch):
        train_loss = 0.0
        train_score=0
        self.model.train()
        start_time = time.time()
        #with tqdm(total=self.epoch_size_train,desc=f'Epoch {epoch + 1}/{self.n_epochs}',postfix=dict,mininterval=0.3) as pbar:

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

            waste_time = time.time() - start_time
            op_lr=get_lr(self.optimizer)

            tbar.set_postfix(**{'train_loss' : train_loss / (iteration + 1),
                                'train_score' : train_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : op_lr})
            start_time = time.time()
        a_t_loss=train_loss / len(self.train_dataset)
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
                #score     = self.accuracy(prediction, label)
                score=2-loss.item()
                val_loss += loss.item()
                val_score+=score
            tbar.set_postfix(**{'val_loss' : val_loss / (iteration + 1),
                                'val_score': val_score / (iteration + 1)})
        t_loss=val_loss / len(self.val_dataset)

        if t_loss<self.best_loss:
            self.best_loss=t_loss
            self.best_loss_list.append(self.best_loss)
            if len(self.best_loss_list)>1:
                last_pth=os.path.join(self.opt.save_dir.save_pth_dir,"val_loss_%.4f.pth"%(self.best_loss_list[-2]))
                print("Last:",last_pth)
                print("New:",os.path.join(self.opt.save_dir.save_pth_dir,"val_loss_%.4f.pth"%(self.best_loss)))
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
        plt.plot(self.train_loss_list, 'b', label="Train loss")
        plt.plot(self.val_loss_list,   'r', label="Validation loss")
        plt.title('Learning loss')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"loss.jpg"))

        plt.figure(1)
        plt.plot(self.train_acc_list, 'b', label="Train acc")
        plt.plot(self.val_acc_list, 'r', label="Validation acc")
        plt.title('Learning acc')
        plt.xlabel("Epoch")
        plt.ylabel("accuracy")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"acc.jpg"))

        plt.figure(2)
        plt.plot(self.lr_list, 'b', label="Lr")
        plt.title('Learning Lr')
        plt.xlabel("Epoch")
        plt.ylabel("Lr")
        plt.savefig(os.path.join(self.opt.save_dir.save_loss_dir,"lr.jpg"))

        #plt.show()

if __name__ == "__main__":
    json_filename=r"../json_configs/kits_2d_unet.json"
    opts=json_dict(json_filename)
    trainer = Trainer(opts)
    for i in range(opts.train.n_epochs):
        trainer.training(i)
        trainer.validation(i)
    trainer.save_plt()

    """
tx=0
for iteration, (seg_data, label) in enumerate(trainer.train_dataset):
    print("train shape:",seg_data.shape,label.shape)
    tx=tx+1
    if tx>3:
        break
    pass

da,la=trainer.val_data_calss[322]
print(da.shape)
print(la.shape)
da=da.squeeze(0)
d=da.numpy()
d=d*255
d=d.astype(np.uint8)
la=la.numpy().astype(np.uint8)

la[la==1]=125
la[la==2]=255
plt.figure(0)
plt.imshow(d)
plt.figure(1)
plt.imshow(la)
plt.show()
"""


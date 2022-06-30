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
from utils.dataloader import  deeplab_dataset_collate,json_file,get_dataset
from utils.metrics import *
from nets.get_net import get_network

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net,epoch,train_dataset,val_dataset,optimizers,json_opts):
    net = net.train()
    total_loss = 0
    total_f_score = 0

    val_toal_loss = 0
    val_total_f_score = 0
    start_time = time.time()

    epoch_size_train = len(train_dataset) // json_opts.train.batchSize
    epoch_size_val   = len(val_dataset) // json_opts.train.batchSize

    with tqdm(total=epoch_size_train,desc=f'Epoch {epoch + 1}/{json_opts.train.n_epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(train_dataset):
            if iteration >= epoch_size_train:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if json_opts.train.use_gpu:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
            optimizers.zero_grad()
            outputs = net(imgs)

            if True:#dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizers.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score'   : total_f_score / (iteration + 1),
                                's/step'    : waste_time,
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{json_opts.train.n_epochs}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_dataset):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = torch.from_numpy(imgs).type(torch.FloatTensor)
                pngs = torch.from_numpy(pngs).type(torch.FloatTensor).long()
                labels = torch.from_numpy(labels).type(torch.FloatTensor)
                if json_opts.train.use_gpu:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

                outputs  = net(imgs)
                # val_loss = CE_Loss(outputs, pngs, num_classes = NUM_CLASSES)
                if True:#
                    main_dice = Dice_loss(outputs, labels)
                    val_loss  = val_loss + main_dice
                #-------------------------------#
                #   计算f_score
                #-------------------------------#
                _f_score = f_score(outputs, labels)

                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score'   : val_total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    loss_history.append_loss(total_loss/(epoch_size_train+1), val_toal_loss/(epoch_size_val+1))
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(json_opts.train.n_epochs))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size_train+1),val_toal_loss/(epoch_size_val+1)))

    print('Saving state, iter:', str(epoch+1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth'%((epoch+1),total_loss/(epoch_size_train+1),val_toal_loss/(epoch_size_val+1)))

if __name__ == "__main__":
    json_filename=r"D:\torch_keras_code\KiTs19_seg\json_configs\kits_2d_unet.json"
    json_opts    = json_file(json_filename)


    model        = get_network(json_opts)
    loss_history = LossHistory("logs/")

    device = "cpu"
    if json_opts.train.use_gpu:
        device = "cuda"
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.to(device)

    if json_opts.model.continue_train:
        model_dict = model.state_dict()
        model_path = json_opts.model.path_pre_model
        print("path:",model_path)
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    optimizer       = get_optimizer(model.parameters(),json_opts)
    lr_scheduler    = get_scheduler(optimizer,json_opts)
    train_loss      = get_criterion(json_opts)

    data_class   = get_dataset(json_opts.train.dataset_name)
    train_data   = data_class("train",json_opts)
    val_data     = data_class("val",json_opts)

    train_dataset= DataLoader(train_data, batch_size=json_opts.train.batchSize, num_workers=json_opts.train.num_workers,
                              pin_memory=True,drop_last=True)
    val_dataset  = DataLoader(val_data, batch_size=json_opts.train.batchSize, num_workers=json_opts.train.num_workers,
                              pin_memory=True, drop_last=True)

    for epoch in range(json_opts.train.n_epochs):
        fit_one_epoch(model,epoch,train_dataset,val_dataset,optimizer,json_opts)
        lr_scheduler.step()


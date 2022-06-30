import os
import logging
import pprint

from tqdm import tqdm
from torch.nn import Parameter
import torch
from torch import nn
from torch.utils import data
import torchvision.transforms as transform
from option import Options

import sys
sys.path.insert(0, '../../')

from mictnet.models import get_classification_model
from mictnet.datasets import get_classification_dataset
from mictnet import utils


class Trainer:
    def __init__(self, args):
        self.args = args
        self.logger, self.console, self.output_dir = utils.file.create_logger(args, 'train')
        self.logger.info(pprint.pformat(args))

        # copy model file
        this_dir = os.path.dirname(__file__)


        device = 'cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu'
        print('Compute device: ' + device)
        self.device = torch.device(device)

        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])

        # dataset
        data_kwargs = {'logger': self.logger, 'transform': input_transform,
                       'base_size': args.base_size, 'crop_size': args.crop_size,
                       'crop_vid': args.crop_vid, 'split': args.split,
                       'root': args.data_folder}
        trainset = get_classification_dataset(args.dataset, mode='train', **data_kwargs)
        testset = get_classification_dataset(args.dataset, mode='val', **data_kwargs)

        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': False} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.n_classes = trainset.n_classes
        # model
        model_kwargs = {'backbone': args.backbone, 'dropout': args.dropout,
                        'version': args.version} \
            if args.model == 'mictresnet' else {}
        self.model = get_classification_model(args.model, pretrained=args.pretrained,
                                              **model_kwargs)
        #self.logger.info(pprint.pformat(self.model))

        # count parameter number
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info("Total number of parameters: %d" % total_params)

        # optimizer
        params_list = [{'params': self.model.parameters(), 'lr': args.lr}, ]
        self.optimizer = torch.optim.SGD(params_list,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # define loss function (criterion)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.model.to(self.device)

        self.best_pred = 0.0

        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
            self.best_pred = 0.0

        # lr scheduler
        self.scheduler = utils.LRScheduler(self.logger, args.lr_scheduler, args.lr,
                                           args.epochs, len(self.trainloader),
                                           lr_step=args.lr_step)


    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        top1 = utils.AverageMeter('acc@1', ':6.2f')
        top5 = utils.AverageMeter('acc@5', ':6.2f')
        tbar = tqdm(self.trainloader)

        for i, (data, target) in enumerate(tbar):
            data = data.to(self.device)
            target = target.to(self.device)
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            pred = self.model(data)
            loss = self.criterion(pred, target)
            loss.backward()
            self.optimizer.step()

            acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
            top1.update(acc1[0], args.batch_size)
            top5.update(acc5[0], args.batch_size)
            train_loss += loss.item()
            tbar.set_description(
                'train_loss: %.3f, acc1: %.3f, acc5: %.3f' %
                (train_loss / (i + 1), top1.avg, top5.avg))


        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)

    def validation(self, epoch):
        val_loss = 0.0
        self.model.eval()
        top1 = utils.AverageMeter('acc@1', ':6.2f')
        top5 = utils.AverageMeter('acc@5', ':6.2f')
        tbar = tqdm(self.valloader, desc='\r')

        for i, (data, target) in enumerate(tbar):
            data = data.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                pred = self.model(data)
                loss = self.criterion(pred, target)
                acc1, acc5 = utils.accuracy(pred, target, topk=(1, 5))
                top1.update(acc1[0], args.batch_size)
                top5.update(acc5[0], args.batch_size)
                val_loss += loss.item()
            tbar.set_description(
                'val_loss:   %.3f, acc1: %.3f, acc5: %.3f' %
                (val_loss / (i + 1), top1.avg, top5.avg))

        new_pred = (top1.avg + top5.avg) / 2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best)


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        if not args.no_val:
            trainer.validation(epoch)



import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from .Add_Optimizer import AdaBound,AdaBelief,AdaBoundW
from .Loss import *

def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs,threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def get_optimizer(params,opt):

    opt_alg = opt.train.optim
    if opt_alg == 'sgd':
        optimizer = optim.SGD(params,
                              lr=opt.Option.sgd.lr_rate,
                              momentum=0.9,
                              nesterov=True,
                              weight_decay=opt.l2_reg_weight)
        return optimizer
    elif opt_alg == 'A_sgd':
        optimizer = optim.ASGD(params,
                               lr=opt.Option.A_sgd.lr_rate,
                               lambd=opt.Option.A_sgd.Lambda,
                               alpha=opt.Option.A_sgd.alpha)
        return optimizer
    elif opt_alg == 'RMS':
        optimizer = optim.RMSprop(params,
                                  lr=opt.Option.RMS.lr_rate,
                                  lambd=opt.Option.RMS.Lambda,
                                  alpha=opt.Option.RMS.alpha)
        return optimizer
    elif opt_alg == 'Adadelta':
        optimizer = optim.Adadelta(params,
                                   lr=opt.Option.Adadelta.lr_rate,
                                   rho=opt.Option.Adadelta.rho)
        return optimizer

    elif opt_alg == 'Adam':
        optimizer = optim.Adam(params,
                               lr=opt.Option.Adam.lr_rate,
                               betas=(0.9, 0.999))
        return optimizer

    elif opt_alg == 'Adamx':
        optimizer = optim.Adamax(params,
                               lr=opt.Option.Adamx.lr_rate,
                               betas=(0.9, 0.999),
                               weight_decay=opt.Option.Adamx.weight_decay)
        return optimizer

    elif opt_alg == 'AdaBound':
        optimizer = AdaBound(params,
                             lr=opt.Option.AdaBound.lr_rate,
                             betas=(0.9, 0.999),
                             weight_decay=opt.Option.AdaBound.weight_decay)
        return optimizer
    elif opt_alg == 'AdaBelief':
        optimizer = AdaBelief(params,
                              lr=opt.Option.AdaBelief.lr_rate,
                              betas=(0.9, 0.999),
                              weight_decay=opt.Option.AdaBelief.weight_decay)
        return optimizer
    elif opt_alg == 'AdaBoundW':
        optimizer = AdaBoundW(params,
                              lr=opt.Option.AdaBelief.lr_rate,
                              betas=(0.9, 0.999),
                              weight_decay=opt.Option.AdaBelief.weight_decay)
        return optimizer
    else:
        raise "Not optimer!!"

def get_criterion(opts):
    criter=opts.train.criterion
    if criter == 'soft_dice_loss':
        criterion = SoftDiceLoss(opts.model.output_nc).cuda()
        return criterion
    elif criter == 'dice_loss_pancreas_only':
        criterion = CustomSoftDiceLoss(opts.model.output_nc, class_ids=[0, 2]).cuda()
        return criterion
    elif criter == 'dice_loss_2':
        criterion = DiceLoss_2(opts).cuda()
        return criterion
    elif criter == 'dice_loss_3':
        criterion = DiceLoss_3(opts).cuda()
        return criterion
    elif criter == 'dice_loss_two_seg':
        criterion = DiceLoss_two_seg(opts).cuda()
        return criterion

def get_scheduler(optimizer, opt):
    print('lr_policy = {}'.format(opt.train.lr_policy))
    lr_policy=opt.train.lr_policy
    if lr_policy == 'Lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        return scheduler
    elif lr_policy == 'CyclicLR':
        scheduler = lr_scheduler.CyclicLR(optimizer,
                                          base_lr=opt.lr_scheduler.CyclicLR.base_lr,
                                          max_lr=opt.lr_scheduler.CyclicLR.max_lr,
                                          step_size_up=opt.lr_scheduler.CyclicLR.step_size_up,
                                          step_size_down=opt.lr_scheduler.CyclicLR.step_size_down,
                                          mode=opt.lr_scheduler.CyclicLR.mode,
                                          gamma=opt.lr_scheduler.CyclicLR.gamma,
                                          cycle_momentum=False)
        return scheduler
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_scheduler.step.step_size,
                                        gamma=opt.lr_scheduler.step.gamma)
        return scheduler
    elif lr_policy == 'multi_step':
        scheduler = lr_scheduler.MultiStepLR(optimizer,
                                             milestones=opt.lr_scheduler.multi_step.milestones,
                                             gamma=opt.lr_scheduler.multi_step.gamma)
        return scheduler
    elif lr_policy == 'cosine_Anneal':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=opt.lr_scheduler.cosine_Anneal.T_max,
                                                   eta_min=opt.lr_scheduler.cosine_Anneal.eta_min)
        return scheduler
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode=opt.lr_scheduler.plateau.mode,
                                                   factor=opt.lr_scheduler.plateau.factor,
                                                   threshold=opt.lr_scheduler.plateau.threshold,
                                                   patience=opt.lr_scheduler.plateau.patience,
                                                   verbose=opt.lr_scheduler.plateau.verbose,
                                                   min_lr=opt.lr_scheduler.plateau.min_lr)
        return scheduler
    elif lr_policy == 'exponential':
        scheduler = lr_scheduler.ExponentialLR(optimizer,
                                               gamma=opt.lr_scheduler.exponential.gamma)

        return scheduler

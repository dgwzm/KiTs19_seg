import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import sys
def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

class DiceLoss_2(nn.Module):
    def __init__(self,opts):
        super(DiceLoss_2, self).__init__()
        self.smooth=0.0001
        self.n_classes=2
        wh=256
        #if opts !=None:
        #    wh=opts.wh
        self.one_hot_encoder=Get_One_Hot(opts.wh)

    def forward(self, inputs, target):
        batch_size=target.size(0)
        target_list=self.one_hot_encoder(target)
        target_flat = target_list.view(batch_size, self.n_classes, -1)
        input_flat = inputs.view(batch_size, self.n_classes, -1)
        intersec = input_flat * target_flat
        loss = (2 * torch.sum(intersec, 2) + self.smooth) / (torch.sum(input_flat, 2) + torch.sum(target_flat, 2) + self.smooth)
        loss[:,0]=loss[:,0] * 0.5
        #loss[:, 1] = loss[:, 1] * 0.5
        #print("loss shape ",loss.shape)  # 14, 2
        #print("loss  ", loss)
        loss_sum = torch.sum(loss[:,1])
        # print("loss_sum",loss_sum)

        loss_s=1-loss_sum/batch_size
        #los_s=loss_s/2
        return loss_s

class DiceLoss_3(nn.Module):
    def __init__(self,opts):
        super(DiceLoss_3, self).__init__()
        self.smooth=0.001
        self.n_classes=3
        wh=256
        #if opts !=None:
        #    wh=opts.wh
        self.one_hot_encoder=Get_One_Hot(opts.wh,depth=3)

    def forward(self, inputs, target):
        batch_size=target.size(0)
        target_list=self.one_hot_encoder(target)
        target_flat = target_list.view(batch_size, self.n_classes, -1)
        input_flat = inputs.view(batch_size, self.n_classes, -1)
        intersec = input_flat * target_flat
        loss = (2 * torch.sum(intersec, 2) + self.smooth) / (torch.sum(input_flat, 2) + torch.sum(target_flat, 2) + self.smooth)
        #print("loss shape ",loss.shape)  # 14, 2
        #print("loss  ", loss)
        #loss[:,0]=loss[:,0]*0.01
        #loss[:,1]=loss[:,1]*0.495
        #loss[:,2]=loss[:,2]*0.495
        
        #loss_sum = torch.sum(loss)
        # print("loss_sum",loss_sum)
        #loss_s=1-loss_sum/(batch_size*3)
        
        loss_1=torch.sum(loss[:,1])
        loss_2=torch.sum(loss[:,2])
        Loss_1=1-loss_1/batch_size
        Loss_2=1-loss_2/batch_size

        #loss_s=1-loss_sum/(batch_size*2)
        loss_s=Loss_1+Loss_2
        return Loss_1

class DiceLoss_two_seg(nn.Module):
    def __init__(self,opts):
        super(DiceLoss_two_seg, self).__init__()
        self.smooth=0.001
        self.n_classes=2
        wh=256
        #if opts !=None:
        #    wh=opts.wh
        self.one_hot_encoder=Get_One_Hot(opts.wh,depth=2)

    def forward(self, inputs, target):
        batch_size=target.size(0)
        filver=target.cpu().detach().numpy()
        cancer=target.cpu().detach().numpy()
        filver[filver==2]=1
        cancer[cancer==1]=0
        cancer[cancer==2]=1
        filver=Variable(torch.from_numpy(filver)).cuda()
        cancer=Variable(torch.from_numpy(cancer)).cuda()

        filver_list=self.one_hot_encoder(filver)
        filver_flat = filver_list.view(batch_size, self.n_classes, -1)
        input_flat = inputs[0].view(batch_size, self.n_classes, -1)
        intersec = input_flat * filver_flat
        filver_loss = (2 * torch.sum(intersec, 2) + self.smooth) / (torch.sum(input_flat, 2) + torch.sum(filver_flat, 2) + self.smooth)
        
        cancer_list=self.one_hot_encoder(cancer)
        cancer_flat = cancer_list.view(batch_size, self.n_classes, -1)
        input_flat = inputs[1].view(batch_size, self.n_classes, -1)
        intersec = input_flat * cancer_flat
        cancer_loss = (2 * torch.sum(intersec, 2) + self.smooth) / (torch.sum(input_flat, 2) + torch.sum(cancer_flat, 2) + self.smooth)
        
        #print("loss shape ",loss.shape)  # 14, 2
        #print("loss  ", loss)
        #loss[:,0]=loss[:,0]*0.01
        #loss[:,1]=loss[:,1]*0.495
        #loss[:,2]=loss[:,2]*0.495
        
        #loss_sum = torch.sum(loss)
        # print("loss_sum",loss_sum)
        #loss_s=1-loss_sum/(batch_size*3)
        
        loss_1=torch.sum(filver_loss[:,1])
        loss_2=torch.sum(cancer_loss[:,1])
        Loss_1=1-loss_1/batch_size
        Loss_2=1-loss_2/batch_size

        #loss_s=1-loss_sum/(batch_size*2)
        loss_s=Loss_1+Loss_2
        return loss_s

class SoftDiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
  
    def forward(self, input, target):
        smooth = 0.000000001
        batch_size = input.size(0)
  
        input = F.softmax(input, dim=1).view(batch_size, self.n_classes, -1)
        #input = input.view(batch_size, self.n_classes, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
  
        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth
  
        tor_sum=2.0 * inter / union
        #print("inter", inter.shape, "union", union.shape,"tor_sum",tor_sum.shape)
        #print("tor_sum", tor_sum)
        tor_sum[0][0]=tor_sum[0][0]*0.78
        tor_sum[0][1] = tor_sum[0][1] * 0.22
        #print("tor_sum", tor_sum)
        #score = torch.sum(2.0 * inter / union)
        score = torch.sum(tor_sum)
        score = 1.0 - score / (float(batch_size))
        #* float(self.n_classes))

        return score

class CustomSoftDiceLoss(nn.Module):
    def __init__(self, n_classes, class_ids):
        super(CustomSoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_classes).forward
        self.n_classes = n_classes
        self.class_ids = class_ids

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)

        input = F.softmax(input[:,self.class_ids], dim=1).view(batch_size, len(self.class_ids), -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_classes, -1)
        target = target[:, self.class_ids, :]

        inter = torch.sum(input * target, 2) + smooth
        union = torch.sum(input, 2) + torch.sum(target, 2) + smooth

        score = torch.sum(2.0 * inter / union)
        score = 1.0 - score / (float(batch_size) * float(self.n_classes))

        return score

class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, X_in):
        n_dim = X_in.dim()
        output_size = X_in.size() + torch.Size([self.depth])
        num_element = X_in.numel()
        X_in = X_in.data.long().view(num_element)
        out = Variable(self.ones.index_select(0, X_in)).view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

class Get_One_Hot(nn.Module):
    def __init__(self,wh,depth=2):
        super(Get_One_Hot, self).__init__()
        self.wh=wh
        self.ones = torch.sparse.torch.eye(depth).cuda()

    def forward(self, label):
        batch=label.size(0)
        label=label.view(-1).long()
        ones=self.ones.index_select(0,label)
        ones=torch.transpose(ones,0,1)
        one_high=ones.shape[0]
        label_list=None
        for i in range(one_high):
            if i ==0:
                label_list=ones[i].reshape(batch,1,self.wh,self.wh)
            else:
                label_list=torch.cat((label_list,ones[i].reshape(batch,1,self.wh,self.wh)),1)
            #label_list.append(ones[i].reshape(batch,256,256))
        return label_list


if __name__ == '__main__':
    from torch.autograd import Variable
    depth=3
    batch_size=2
    encoder = One_Hot(depth=depth).forward
    y = Variable(torch.LongTensor(batch_size, 1, 1, 2 ,2).random_() % depth).cuda()  # 4 classes,1x3x3 img
    y_onehot = encoder(y)
    x = Variable(torch.randn(y_onehot.size()).float()).cuda()
    dicemetric = SoftDiceLoss(n_classes=depth)
    dicemetric(x,y)

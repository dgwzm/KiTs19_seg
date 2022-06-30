import torch
import torch.nn as nn
import torch.nn.functional as F

x=torch.randn(2,1,3,2)
print(x)
print(torch.sigmoid(x))
#print(torch.softmax(x,dim=1))

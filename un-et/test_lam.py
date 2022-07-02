import math
import matplotlib.pyplot as plt
import random
max=0
min=0
for i in range(1000):
    a = random.uniform(0,1)
    if a<0.6:
        min=min+1
    else:
        max=max+1
print(max,min)

def lambda_rule(epoch):
    lr_l = (((1 + math.cos(epoch * math.pi / 150)) / 2) ** 1.0) * 0.8 + 0.1
    return lr_l

list_d=[]
for i in range(150):
    s=lambda_rule(i)*0.001
    list_d.append(s)
plt.plot(list_d)
plt.show()
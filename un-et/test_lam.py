import math
import matplotlib.pyplot as plt
import random

import datetime

now = datetime.datetime.now()

otherStyleTime = now.strftime("%Y-%m-%d_%H:%M:%S")
print(otherStyleTime)
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
    lr_l = (((1 + math.cos(epoch * math.pi / 200)) / 2) ** 1.0) * 0.8 + 0.1
    return lr_l
def lambda_rule_2(epoch):
    lr_l = (((1 + math.cos(epoch * math.pi / 200)) / 2) ** 1.5) * 0.8 + 0.1
    return lr_l

list_d=[]
list_2=[]
for i in range(200):
    s=lambda_rule(i)*0.001
    s_1=lambda_rule_2(i)*0.001
    list_d.append(s)
    list_2.append(s_1)

plt.plot(list_d)
plt.plot(list_2)
plt.legend(["train","val"])
plt.show()
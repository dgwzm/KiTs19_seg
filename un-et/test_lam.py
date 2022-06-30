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

# def lambda_rule(epoch):
#     lr_l = (0.02+math.cos((epoch*math.pi/100)**2)/2)*0.09+0.1
#     return lr_l
#
# list_d=[]
# for i in range(100):
#     s=lambda_rule(i)
#     list_d.append(s)
# plt.plot(list_d)
# plt.show()
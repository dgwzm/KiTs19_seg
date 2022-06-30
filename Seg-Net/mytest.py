import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
p=r"D:\torch_keras_code\kits_data_label\label_jpg\segmentation_00000\298.png"
i=cv2.imread(p)
print(i.shape)
i=i[:,:,0]
d=np.array(i)
ar_list=[]
nm=0
for x in d:
    for i in x:
        if i not in [0,125,255]:
            nm=nm+1
            ar_list.append(i)
print(nm)
plt.figure(0)
plt.imshow(d)

# new_img=np.zeros(d.shape)
# print(new_img.shape)
#
# for i in range(512):
#     for j in range(512):
#         if d[i,j] <100:
#             new_img[i,j]=0
#         elif d[i,j] >=100 and d[i,j] <200:
#             new_img[i,j]=125
#         elif d[i,j] >=200 :
#             new_img[i,j]=255
#
# plt.figure(1)
# plt.imshow(new_img)
# cv2.imwrite(r"D:\torch_keras_code\kits_data_label\322.png",new_img)
# #plt.savefig(r"D:\torch_keras_code\kits_data_label\322.png")
plt.show()

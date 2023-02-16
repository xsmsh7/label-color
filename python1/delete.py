#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 17:05:45 2023

@author: intern
"""

import os,sys
import glob
from PIL import Image
import numpy as np

Img2_dir = 'C:/Users/DWEI/Downloads/Falldown_PPE.v1i.yolov7pytorch/train/images'
Img1_dir = 'C:/Users/DWEI/Downloads/Falldown_PPE.v2i.yolov7pytorch/train/images'

img1_Lists = os.listdir(Img1_dir)
img2_Lists = os.listdir(Img2_dir)
print('version1 : ',np.size(img2_Lists))
print('version2 : ',np.size(img1_Lists))
img2_names = []
for item in img2_Lists:
    img2_names.append(os.path.basename(item))

print('version1 name : ',np.size(img2_names))

n = 0
for item in img2_names:
    img1namespath = os.path.join(Img1_dir + "/" + item)
    #print(item)
    #print(img1namespath)
    if os.path.exists(img1namespath):
        n = n + 1
        os.remove(img1namespath)
        print('刪除文件:',item)

       
print('共計刪除:',n,'個')
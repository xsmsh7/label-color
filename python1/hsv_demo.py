#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 10:19:56 2023

@author: intern
"""

import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

#%%
img = PIL.Image.open('2.png')
im0 = np.array(img)
plt.imshow(im0[:,:,0:3])
#%%
img1 = PIL.Image.open('1.png')
im1 = np.array(img1)
plt.imshow(im1[:,:,0:3])
#%%
import colorsys
colorsys.rgb_to_hsv(im[:,:,0],im[:,:,1],im[:,:,2])
#%%
def rgb_to_hsv(im):
    
    r, g, b = im[:,:,0]/255.0, im[:,:,1]/255.0, im[:,:,2]/255.0
    mx = im[:,:,0:3].max(axis=2)/255.0
    mn = im[:,:,0:3].min(axis=2)/255.0
    df = mx-mn
    
    h = s = np.zeros(mx.shape)
    h[mx == r] = ((60 * ((g-b)/df) + 360) % 360)[mx == r]
    h[mx == g] = ((60 * ((b-r)/df) + 120) % 360)[mx == g]
    h[mx == b] = ((60 * ((r-g)/df) + 240) % 360)[mx == b]
    h[mx == mn] = 0
    
    s = (df/mx)*255
    s[mx == 0] = 0
    v = mx*255
    height = im[:,:,0].shape[0]
    weight = im[:,:,0].shape[1]
    hsvim = np.stack([h, s, v], axis=2).reshape((height, weight, 3))
    hsvim = np.round(hsvim).astype(int)
    return hsvim

hsvim = rgb_to_hsv(im0[:,:,0:3])

plt.imshow(hsvim[:,:,2])
plt.imshow(hsvim)
plt.show()
#%%
img = PIL.Image.open('nobg1.png')
im0 = np.array(img)
plt.imshow(im0[:,:,0:3])
plt.show()
hsvim = rgb_to_hsv(im0[:,:,0:3])
plt.imshow(hsvim)
plt.show()
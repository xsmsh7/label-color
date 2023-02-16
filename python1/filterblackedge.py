#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 14:49:52 2023

@author: intern
"""
 
import cv2

kernel = np.ones((3, 3), dtype=np.uint8)
erosion = cv2.erode(im0, kernel, iterations=1)
plt.imshow( erosion[:,:,0:3])
#%%
erosion = cv2.morphologyEx(im0, cv2.MORPH_OPEN, kernel, 1)
plt.imshow( erosion[:,:,0:3])
hsvim = rgb_to_hsv(erosion[:,:,0:3])
#%%
float("0.5555554573")
a = format(float("0.5555554573"), '.6f')
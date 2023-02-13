#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:55:54 2023

@author: intern
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 
 
# Creating dataset
z = hsvim[:,:,2]
x = hsvim[:,:,0]
y = hsvim[:,:,1]
 
# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines
ax.grid(visible = True, color ='grey',
        linestyle ='-.', linewidth = 0.3,
        alpha = 0.1)
 
 
# Creating color map
my_cmap = plt.get_cmap('hsv')
 
# Creating plot
sctt = ax.scatter3D(x, y, z,
                    alpha = 0.8,
                    c = (x + y + z),
                    cmap = my_cmap,
                    marker ='^')

# plotting

ax.set_title('3D hsv')
ax.set_xlabel('h-axis', fontweight ='bold')
ax.set_ylabel('s-axis', fontweight ='bold')
ax.set_zlabel('v-axis', fontweight ='bold')
plt.show()
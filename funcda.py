#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:47:14 2023

@author: intern
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:44:04 2023

@author: intern
"""


import torch
from IPython.display import Image  # for displaying images
import os 
import random
import shutil
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET
from xml.dom import minidom
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image


#%%image and label show
class_id_to_name_mapping = {0: '0', 1: '1', 2: '2', 3: '3',
                            4:"4" ,5:"5" ,6:"6"}
def plot_bounding_box(image, annotation_list):

    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)
    
    if annotation_list == []:
        plt.imshow(np.array(image))
        return
    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = (transformed_annotations[:,1] 
        - (transformed_annotations[:,3] / 2))
    transformed_annotations[:,2] = (transformed_annotations[:,2] 
        - (transformed_annotations[:,4] / 2))
    transformed_annotations[:,3] = (transformed_annotations[:,1] 
        + transformed_annotations[:,3])
    transformed_annotations[:,4] = (transformed_annotations[:,2] 
        + transformed_annotations[:,4])

    font = ImageFont.truetype(
        '/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 100)

    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        number = int(obj_cls)
        plotted_image.rectangle(
            ((x0,y0), (x1,y1)), 
            outline=(0+number*60, 0+number*120, 200+number*180), width=10)
        plotted_image.text(
            (x0, y0 - 50), class_id_to_name_mapping[number], 
            fill="black", font = font)
    
    #plt.imshow(image)
    plt.show()
    
    return image
#%%label show
def plot_label(image, annotation_list):
    
    data = {}
    annotations = np.array(annotation_list)
    w, h = image.size
    
    plotted_image = ImageDraw.Draw(image)
    
    if annotation_list == []:
        plt.imshow(np.array(image))
        return
    transformed_annotations = np.copy(annotations)
    transformed_annotations[:,[1,3]] = annotations[:,[1,3]] * w
    transformed_annotations[:,[2,4]] = annotations[:,[2,4]] * h 
    
    transformed_annotations[:,1] = (transformed_annotations[:,1] 
        - (transformed_annotations[:,3] / 2))
    transformed_annotations[:,2] = (transformed_annotations[:,2] 
        - (transformed_annotations[:,4] / 2))
    transformed_annotations[:,3] = (transformed_annotations[:,1] 
        + transformed_annotations[:,3])
    transformed_annotations[:,4] = (transformed_annotations[:,2] 
        + transformed_annotations[:,4])
    transformed_annotations = np.round(transformed_annotations).astype(int)
    
    i=0
    ans = []
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        number = int(obj_cls)
        label_image = np.array(image)[y0:y1 , x0:x1]
        #plt.imshow(label_image)
        plt.show()
        PIL_image = Image.fromarray(np.uint8(label_image)).convert('RGB')
        i = i+1
        if obj_cls == 2:
            ans.append([i,PIL_image])
            
    plt.show()
    
    return ans
#%%remove background
def remove_background(PIL_image):
    
    output = remove(PIL_image)
    #plt.imshow(output)
    plt.show()
    
    return output
#%%remove transparent
def remove_transparent(PIL_image : Image):
    PIL_image = np.array(PIL_image)
    PIL_image[PIL_image[:,:,3] < 60] = 0
    
    return PIL_image
#%%rgb to hsv
def rgb_to_hsv(im):
    
    r, g, b = im[:,:,0]/255.0, im[:,:,1]/255.0, im[:,:,2]/255.0
    mx = im[:,:,0:3].max(axis=2)/255.0
    mn = im[:,:,0:3].min(axis=2)/255.0
    df = mx-mn
    
    h = s = np.zeros(mx.shape)
    h[([mx == r] and [df != 0])[0]] = ((60 * ((g-b)[([mx == r] and [df != 0])[0]]/df[([mx == r] and [df != 0])[0]]) + 360) % 360)
    h[([mx == g] and [df != 0])[0]] = ((60 * ((b-r)[([mx == g] and [df != 0])[0]]/df[([mx == g] and [df != 0])[0]]) + 120) % 360)
    h[([mx == b] and [df != 0])[0]] = ((60 * ((r-g)[([mx == b] and [df != 0])[0]]/df[([mx == b] and [df != 0])[0]]) + 240) % 360)
    h[mx == mn] = 0
    
    s[mx != 0] = (df[mx != 0]/mx[mx != 0])*255
    v = mx*255
    height = im[:,:,0].shape[0]
    weight = im[:,:,0].shape[1]
    hsvim = np.stack([h, s, v], axis=2).reshape((height, weight, 3))
    hsvim = np.round(hsvim).astype(int)
    
    return hsvim
#%%color kmean
from sklearn.cluster import KMeans

def image_color_separate(hsv : np.array):
    
    height = hsv.shape[0]
    weight = hsv.shape[1]
    
    X = hsv.reshape((height * weight, 3))
    X = X[np.all(X,axis=1),:]
    
    kmeans = KMeans(n_clusters=3, n_init = 'auto')
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    centers = np.array(kmeans.cluster_centers_)
    number_weight = (np.array([(y_kmeans==0).sum(), (y_kmeans==1).sum(), 
                              (y_kmeans==2).sum()]) / len(y_kmeans))
    return centers,number_weight
#%%loss_function
def loss_function(goal_color : np.array, detecter : np.array):#one center
    ans = 'none'
    detect_color = np.array(detecter)
    i = 1
    if (goal_color[0][1,0] > detecter[0] > goal_color[0][0,0]) and (goal_color[0][1,1] > detecter[1]/255 > goal_color[0][0,1]) and (goal_color[0][1,2] > detecter[2]/255 > goal_color[0][0,2]):
        ans = 'blue'
        print('blue')
    elif (goal_color[i][1,0] > detecter[0] > goal_color[i][0,0]) and (goal_color[i][1,1] > detecter[1]/255 > goal_color[i][0,1]) and (goal_color[i][1,2] > detecter[2]/255 > goal_color[i][0,2]):
        ans = 'white'
        print('white') 
    else:
        print('none')
    return ans

#%%color_detect
def color_detect(loss : list, number_weight):#list str
    if number_weight[loss == 'blue'].sum() > 0.5:
        print("Blue")
        ans = "Blue"
    if number_weight[color_list[i] == 'white'].sum() > 0.5:
        print("White")
        ans = "White"
    return ans
#%%
import os

#%%
def change_label(path, annotation_list):
    
    f = open(path, 'w')
    for i in range(0, len(annotation_list)):
        for line in annotation_list[i]:
            f.write(str(line) + " ")
        f.write('\n')
    f.close()
    

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

data = {}

#%%

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
        
    plt.imshow(np.array(image))
    plt.show()
    
    
#%%
def plot_label(image, annotation_list):
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
    
    %cd ./output
    i=0
    data["label_num"] = []
    for ann in transformed_annotations:
        obj_cls, x0, y0, x1, y1 = ann
        number = int(obj_cls)
        label_image = np.array(image)[y0:y1 , x0:x1]
        plt.imshow(label_image)
        plt.show()
        PIL_image = Image.fromarray(np.uint8(label_image)).convert('RGB')
        i = i+1
        output_path = 'label' + str(i) + '.png'
        if obj_cls == 2:
            PIL_image.save(output_path)
            data["label_num"].append(i)
        
    %cd ../
    
    plt.show()

#%% file_manage
Img_dir = '/home/intern/下載/old data/real_PPE-data/train/labels'
annotations = os.listdir(Img_dir)
%cd /home/intern/下載/old data/real_PPE-data/train/labels
#%%
class_id_to_name_mapping = {
        0: '0', 1: '1', 2: '2', 3: '3', 4:"4" ,5:"5" ,6:"6"}
for i in range(0,1):
    annotation_file = annotations[i]
    data['label_file'] = annotation_file
    
    %cd ../labels
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = [[float(y) for y in x ] for x in annotation_list]

    #Get the corresponding image file
    image_file = annotation_file.replace(
        "annotations", "images").replace("txt", "jpg")
    %cd ../images
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)

    #Plot the Bounding Box
    plot_label(image, annotation_list)
    plot_bounding_box(image, annotation_list)

#%% file_manage
Img_label_dir = '/home/intern/下載/old data/real_PPE-data/train/images/output'
%cd /home/intern/下載/old data/real_PPE-data/train/images/output
Img_label_Lists = os.listdir(Img_label_dir)
#%%
for i in range(0,2):
    Img_label = Img_label_Lists[i]
    PIL_image = Image.open(Img_label)
    
    output = remove(PIL_image)
    
    %cd ../nobackground
    output_path = ('/home/intern/下載/old data/real_PPE-data/train/images/'
                   'nobackground/nobg' + str(i) + '.png')
    output.save(output_path)
    %cd ../output
    
    plt.imshow(np.array(output))
    plt.show()
    
#%%
import PIL.Image

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

#%% file_manage
Img_label_dir = ('/home/intern/下載/old data/real_PPE-data/train/images'\
                 '/nobackground')
%cd /home/intern/下載/old data/real_PPE-data/train/images/nobackground
Img_NoBG_Lists = os.listdir(Img_label_dir)
#%%
label_list = []
for i in range(0,2):
    Img_label = Img_NoBG_Lists[i]
    PIL_image = Image.open(Img_label)
    PIL_image = np.array(PIL_image)
    PIL_image[PIL_image[:,:,3] < 60] = 0
    plt.imshow(PIL_image[:,:,0:3])
    plt.show()
    hsvim = rgb_to_hsv(PIL_image[:,:,0:3])
    plt.imshow(hsvim)
    plt.show()
    
    label_list.append(hsvim)
    """%cd ../hsv
    output_path = ('/home/intern/下載/old data/real_PPE-data/train/images/'
                   'hsv/nobg' + str(i) + '.png')
    np.save(output_path, hsvim)
    %cd ../nobackground"""
    
#%%
from sklearn.cluster import KMeans
kmean_list = []
number_weight = []
for i in range(0, len(label_list)):
    height = label_list[i][:,:,0].shape[0]
    weight = label_list[i][:,:,0].shape[1]
    X = label_list[i].reshape((height * weight, 3))
    x = np.all(X,axis=1)
    X = X[x,:]
    
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    
    centers = np.array(kmeans.cluster_centers_)
    kmean_list.append(centers)
    number_weight.append(np.array(
        [(y_kmeans==0).sum(), (y_kmeans==1).sum(), (y_kmeans==2).sum()]) 
        / len(y_kmeans))
#%%
goal = []
goal.append(np.array([[200,0.2,0.2], [250,1,1]]).astype('float64'))
goal.append(np.array([[0,0,0.6], [360,0.2,1]]).astype('float64'))
def loss_function(goal_color, detecter):
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

#%%
color_list = []
for i in range(0, 2):
    color_weight = np.array([])
    for j in range(0, 3):
        color_weight = np.append(color_weight,
                                 loss_function(goal, kmean_list[i][j, :]))
    color_list.append(color_weight)   
    if number_weight[i][color_list[i] == 'blue'].sum() > 0.5:
        print("Blue",i)
    if number_weight[i][color_list[i] == 'white'].sum() > 0.5:
        print("white",i)
        
#%% file_manage
%cd /home/intern/下載/old data/real_PPE-data/train/newlabels
#%%

%cd ../labels
with open(data['label_file'], "r") as file:
    annotation_list = file.read().split("\n")[:-1]
    annotation_list = [x.split(" ") for x in annotation_list]
    annotation_list = [[float(y) for y in x ] for x in annotation_list]  
%cd ../labels
    f = open(item, 'w')annotation_list  
        
annotation_list[data["label_num"][i]][0]
#%%
import os

path = 'output.txt'
f = open(path, 'w')
lines = ['Hello World\n', '123', '456\n', '789\n']
f.writelines(lines)
f.close()
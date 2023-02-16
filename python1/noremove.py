#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:00:52 2023

@author: intern
"""

%cd /home/intern/python1
import funcda as f

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
from rembg import remove
from sklearn.cluster import KMeans

data = {}

goal = []
goal.append(np.array([[200,0.2,0.2], [250,1,1]]).astype('float64'))
goal.append(np.array([[0,0,0.5], [360,0.2,1]]).astype('float64'))
#%% file_manage
Img_dir = '/home/intern/下載/old data/real_PPE-data/test/labels'
annotations = os.listdir(Img_dir)
%cd /home/intern/下載/old data/real_PPE-data/test/labels
#%%
for k in range(0,1):
    
    annotation_file = "220311_1802540000000127.txt"
    #annotation_file = annotations[k]
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
    label_table = f.plot_label(image, annotation_list)
    #f.plot_bounding_box(image, annotation_list)


    for i in range(0, len(label_table)):
        PIL_image = label_table[i][1]
        PIL_image = np.array(PIL_image)
        label_table[i].append(PIL_image)
        
        #rgb to hsv
        hsvim = f.rgb_to_hsv(label_table[i][2])     
        label_table[i].append(hsvim)
        
        
    for i in range(0, len(label_table)):    
        #color kmean
        centers,number_weight = f.image_color_separate(label_table[i][3])
        
        label_table[i].append(centers)
        label_table[i].append(number_weight)

    
    for i in range(0, len(label_table)):
        color_weight = []
        for j in range(0, 3):
            color = f.loss_function(goal, label_table[i][4][j, :])
            color_weight.append(color)
        label_table[i].append(np.array(color_weight))


    for i in range(0, len(label_table)):
        title = "None" + annotation_file + str(label_table[i][0])
        
        if label_table[i][5][label_table[i][6] == 'blue'].sum() > 0.4:
            title = "Blue " + annotation_file + str(label_table[i][0])
            annotation_list[label_table[i][0]-1][0] = 3
            %cd ../newlabels
            f.change_label(annotation_file, annotation_list)
        if label_table[i][5][label_table[i][6] == 'white'].sum() > 0.4:
            title = "White " + annotation_file + str(label_table[i][0])
            annotation_list[label_table[i][0]-1][0] = 4
            %cd ../newlabels
            f.change_label(annotation_file, annotation_list)
        plt.imshow(label_table[i][1])
        plt.title(title)
        plt.show()
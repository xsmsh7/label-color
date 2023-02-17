#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:22:21 2023

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
Img_dir = '/media/intern/6bf406ea-c22f-457e-bdbc-47701a7188a4/home/content/yolov7/olddata/test/labels'
annotations = os.listdir(Img_dir)
%cd /media/intern/6bf406ea-c22f-457e-bdbc-47701a7188a4/home/content/yolov7/olddata/test/labels
#%%
list_bad = []
for k in range(0,80):
    
    annotation_file = annotations[k]
  
    %cd ../labels
    old = r_yolo_txt(annotation_file)
    old_oj = np.asarray(old)[:,0]  
    %cd ../labels1
    new = r_yolo_txt(annotation_file)
    new_oj = np.asarray(new)[:,0]
    if not np.array_equal(old_oj, new_oj):
        list_bad.append(annotation_file)
#%%
a = lambda s: s[0]
def b(s):
    s = s[0]
    return s

#%%
def r_yolo_txt(annotation_file):
    with open(annotation_file, "r") as file:
        annotation_list = file.read().split("\n")[:-1]
        annotation_list = [x.split(" ") for x in annotation_list]
        annotation_list = np.asarray(annotation_list)
        if len(annotation_list[0]) == 6:
            annotation_list = np.delete(annotation_list , 5, axis = 1)
        annotation_list = [[float(y) for y in x ] for x in annotation_list]
        
        annotation_list.sort(key = b)
        return annotation_list
#%%
for k in range(0,20):
    
    annotation_file = list_bad[k]
  
    %cd ../labels
    
    annotation_list = r_yolo_txt(annotation_file)
        
    #Get the corresponding image file
    image_file = annotation_file.replace(
        "annotations", "images").replace("txt", "jpg")
    %cd ../images
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)
    #Plot the Bounding Box
    a = f.plot_bounding_box(image, annotation_list)
    plt.imshow(a)
    plt.show()
    #label_table,a = f.plot_label(image, annotation_list)
    
    %cd ../labels1
    annotation_list = r_yolo_txt(annotation_file)
        
    #Get the corresponding image file
    image_file = annotation_file.replace(
        "annotations", "images").replace("txt", "jpg")
    %cd ../images
    assert os.path.exists(image_file)

    #Load the image
    image = Image.open(image_file)
    #Plot the Bounding Box
    a = f.plot_bounding_box(image, annotation_list)
    plt.imshow(a)
    plt.show()
    #label_table,a = f.plot_label(image, annotation_list)
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:39:35 2023

@author: intern
"""

import shutil
import os
#%% file_manage
Img_dir = '/home/intern/下載/old data/real_PPE-data/test/newlabels'
annotations = os.listdir(Img_dir)
%cd /home/intern/下載/old data/real_PPE-data/test/newlabels
#%%
for k in range(0,1934):
    file = str(annotations[k]).replace("txt", "jpg")
    src = '/home/intern/下載/old data/real_PPE-data/test/images/' + file
    dst = '/home/intern/下載/old data/real_PPE-data/test/images1/' + file
    
    shutil.copyfile(src, dst)
#%%
%cd /home/intern/下載/old data/real_PPE-data/train/newlabels
for k in range(0,100):
    if not os.path.exists(annotations[k]):
        print(k)
        file = str(annotations[k])
        src = '/home/intern/下載/old data/real_PPE-data/train/labels/' + file
        dst = '/home/intern/下載/old data/real_PPE-data/train/newlabels/' + file
        shutil.copyfile(src, dst)
#%%watch no detect
%cd /home/intern/下載/old data/real_PPE-data/test/newlabels
annotation = []
for k in range(0,2388):
    if not os.path.exists(annotations[k]):
        print(k)
        annotation.append(annotations[k])
#%%
for k in range(50,100):
    annotation_file = annotation[k]
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
    label_image = f.plot_label(image, annotation_list)
    for i in range(0, len(label_image)):
        plt.imshow(label_image[i][1])
        plt.title(annotation_file)
        plt.show()
    label_image = f.plot_bounding_box(image, annotation_list)
    plt.imshow(label_image)
    plt.show()
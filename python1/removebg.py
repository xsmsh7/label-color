#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:40:01 2023

@author: intern
"""

from rembg import remove

input_path = 'output1.png'
output_path = 'output1.png'

with open(input_path, 'rb') as i:
    with open(output_path, 'wb') as o:
        input = i.read()
        output = remove(input)
        o.write(output)
        

#%%
image_file = '1.png'
PIL_image = Image.open(image_file)
PIL_image = Image.fromarray(np.uint8(b)).convert('RGB')
#%%
output = remove(PIL_image)
output.save(output_path)
#%%
input = cv2.imread(input_path)
output = remove(input)
cv2.imwrite(output_path, output)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:33:14 2023

@author: intern
"""

import os

path = 'output.txt'
f = open(path, 'w')
for i in range(0, len(annotation_list)):
    f.writelines(str(annotation_list[i])+"\n")
f.close()
#%%
path = 'output.txt'
f = open(path, 'w')
f.write('Hello World')
f.write(123)
f.write(123.45)
f.close()
#%%
lines = ['Readme', 'How to write text files in Python']
with open('readme.txt', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
#%%多個空格
path = 'output.txt'
f = open(path, 'w')
for i in range(0, len(annotation_list)):
    for line in annotation_list[i]:
        f.write(str(line) + " ")
    f.write('\n')
f.close()
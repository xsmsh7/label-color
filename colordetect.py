#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 10:29:48 2023

@author: intern
"""
#An efficient detection method for rare colored capsule based on RGB and HSV color space 
#Distance measures in RGB and HSV color spaces

#%%
goal_color_1 = np.array([20,1,1]).astype('float64')
def loss_function(goal_color, detecter):
    detect_color = np.array(detecter)
    
    Theta = goal_color[0] - detect_color[0]
    detect_color[1] = detect_color[1]/255
    DeltaV = goal_color[2] - detect_color[2]/255
    DeltaC = goal_color[1]**2 + detect_color[1]**2 - 2*goal_color[1]*detect_color[1]*np.deg2rad(np.cos(Theta))
    
    K = 0.2#paper have mistake
    
    DeltaHSV =  (K*DeltaV**2 + (1-K)*DeltaC)**0.5
    return DeltaHSV

#%%
goal_color_1 = np.array([[200,0.2,0.2], [250,1,1]]).astype('float64')
def loss_function(goal_color, detecter):
    ans = False
    detect_color = np.array(detecter)
    if (goal_color[1,0] > detecter[0] > goal_color[0,0]) and (goal_color[1,1] > detecter[1]/255 > goal_color[0,1]) and (goal_color[1,2] > detecter[2]/255 > goal_color[0,2]):
        ans = True
    else:
        print("1")
    return ans
    
#%%
colorweight = np.array([]).astype('bool')
colorweight = np.append(colorweight, loss_function(goal_color_1, centers[0, :]))
colorweight = np.append(colorweight, loss_function(goal_color_1, centers[1, :]))
colorweight = np.append(colorweight, loss_function(goal_color_1, centers[2, :]))
if numberweight[colorweight].sum() > 0.5:
    print("Blue")
#%%
a = 2
def lo(i, a):
    i = i +1
    print( i )
    print( a )
lo(a, 3)
#%%
def loss(goal_color, detecter):
    

    goal_color = goal_color[0:3]*2
    a = goal_color /10
    print( a )
    return

loss(goal_color_1, centers[0, :])
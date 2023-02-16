#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 16:08:43 2023

@author: intern
"""


height = hsvim.shape[0]
weight = hsvim.shape[1]
X = hsvim.reshape((height * weight, 3))
x = np.all(X,axis=1)
X = X[x,:]

#%%
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

#%%
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
ax.scatter3D(X[:, 0], X[:, 1], X[:, 2],alpha = 0.8,
                    marker ='^', c=y_kmeans, s=50, cmap='viridis')
ax.set_xlabel('h-axis', fontweight ='bold')
ax.set_ylabel('s-axis', fontweight ='bold')
ax.set_zlabel('v-axis', fontweight ='bold')
centers = np.array(kmeans.cluster_centers_)
ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
#%%
fig = plt.figure(figsize = (16, 9))
numberweight = np.array([(y_kmeans==0).sum(), (y_kmeans==1).sum(), (y_kmeans==2).sum()]) / y_kmeans.sum() 
ax.scatter3D(centers[:, 0], centers[:, 1], centers[:, 2], c='black', s=200, alpha=0.5)
y_kmeans[y_kmeans= 0]
plt.show()

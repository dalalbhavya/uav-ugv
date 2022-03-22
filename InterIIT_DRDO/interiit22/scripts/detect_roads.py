#!/usr/bin/env python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

import warnings
warnings.filterwarnings("ignore")

import json

from Helpers import *

def get_path(image,t, image_path):
    
    kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
    img_sharp = cv2.filter2D(image, ddepth=-1, kernel=kernel)
    out = cv2.addWeighted( image, 1.1, img_sharp, 0.1, -5)
    squares = split_stitch(out)
    
    masks = []
    for cut_img in squares:
        
        mask = solver.test_one_img_from_path(cut_img)
        mask[mask>0.7] = 255
        mask[mask<=0.7] = 0
        mask = np.concatenate([mask[:,:,None],mask[:,:,None],mask[:,:,None]],axis=2)
        masks.append(mask.astype(np.uint8))
    prediction = split_stitch(image,masks=masks)
    points_path = image_path[0:len(image_path)-4] + ".json"

    points = open(points_path) 
    coords = json.load(points)
    start = np.array(coords["Start"])
    end = np.array(coords["End"])

    out = prediction
    out = cv2.cvtColor(out,cv2.COLOR_BGR2GRAY)
    grid = np.ceil((255-out)/255).astype('int')
    
    edges = (cv2.Canny(out, threshold1=100, threshold2=200))/255
    ngrid = narrowize(grid, edges,t)
    G = initgraph(ngrid)
    start = tuple(start[::-1])
    end = tuple(end[::-1])

    astar_path = generate_path(G,start,end,euclidean)

    return astar_path
    
def submit(): 
    ans = {}
    for id in image_ids:
        image = cv2.imread(source+id)
        paths = get_path(image,4,source+id)
        if paths == None:
            paths = get_path(image,0,source+id)
        path = []
        for point in paths:
            (x,y) = point
            path.append([int(y),int(x)])
        ans[id] = path
        print(id,': Done')
    json_object = json.dumps(ans, indent = 4)
    with open("waypoints.json", "w") as outfile:
        outfile.write(json_object) 


source =  'drive/MyDrive/Test/Data/'
val = os.listdir(source)
image_ids = list()
for filename in val:
  if filename.endswith('.png'):
    image_ids.append(filename)
print(image_ids)    

solver = TTAFrame(DinkNet34)
solver.load('log01_dink34.th')
submit()


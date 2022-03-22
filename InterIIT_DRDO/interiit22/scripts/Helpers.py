import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from numba import njit
from sklearn.neighbors import KDTree

import json


######################################################################################################
# Referred from https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge/blob/master/test.py
# Authors: Lichen Zhou, Chuang Zhang, Ming Wu, Beijing University of Posts and Telecommunications

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

from DeepGlobe.networks.unet import Unet
from DeepGlobe.networks.dunet import Dunet
from DeepGlobe.networks.dinknet import LinkNet34, DinkNet34, DinkNet50, DinkNet101, DinkNet34_less_pool


BATCHSIZE_PER_CARD = 4

class TTAFrame():
    def __init__(self, net):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=range(torch.cuda.device_count()))
        
    def test_one_img_from_path(self, img, evalmode = True):
        if evalmode:
            self.net.eval()
        batchsize = torch.cuda.device_count() * BATCHSIZE_PER_CARD
        if batchsize >= 8:
            return self.test_one_img_from_path_1(img)
        elif batchsize >= 4:
            return self.test_one_img_from_path_2(img)
        elif batchsize >= 2:
            return self.test_one_img_from_path_4(img)

    def test_one_img_from_path_8(self, img):
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2

    def test_one_img_from_path_4(self, img):
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.array(img1)[:,:,::-1]
        img4 = np.array(img2)[:,:,::-1]
        
        img1 = img1.transpose(0,3,1,2)
        img2 = img2.transpose(0,3,1,2)
        img3 = img3.transpose(0,3,1,2)
        img4 = img4.transpose(0,3,1,2)
        
        img1 = V(torch.Tensor(np.array(img1, np.float32)/255.0 * 3.2 -1.6).cuda())
        img2 = V(torch.Tensor(np.array(img2, np.float32)/255.0 * 3.2 -1.6).cuda())
        img3 = V(torch.Tensor(np.array(img3, np.float32)/255.0 * 3.2 -1.6).cuda())
        img4 = V(torch.Tensor(np.array(img4, np.float32)/255.0 * 3.2 -1.6).cuda())
        
        maska = self.net.forward(img1).squeeze().cpu().data.numpy()
        maskb = self.net.forward(img2).squeeze().cpu().data.numpy()
        maskc = self.net.forward(img3).squeeze().cpu().data.numpy()
        maskd = self.net.forward(img4).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,::-1] + maskc[:,:,::-1] + maskd[:,::-1,::-1]
        mask2 = mask1[0] + np.rot90(mask1[1])[::-1,::-1]
        
        return mask2
    
    def test_one_img_from_path_2(self, img):
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = img3.transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        img6 = img4.transpose(0,3,1,2)
        img6 = np.array(img6, np.float32)/255.0 * 3.2 -1.6
        img6 = V(torch.Tensor(img6).cuda())
        
        maska = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        maskb = self.net.forward(img6).squeeze().cpu().data.numpy()
        
        mask1 = maska + maskb[:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3
    
    def test_one_img_from_path_1(self, img):
        #img = cv2.imread(path)#.transpose(2,0,1)[None]
        
        img90 = np.array(np.rot90(img))
        img1 = np.concatenate([img[None],img90[None]])
        img2 = np.array(img1)[:,::-1]
        img3 = np.concatenate([img1,img2])
        img4 = np.array(img3)[:,:,::-1]
        img5 = np.concatenate([img3,img4]).transpose(0,3,1,2)
        img5 = np.array(img5, np.float32)/255.0 * 3.2 -1.6
        img5 = V(torch.Tensor(img5).cuda())
        
        mask = self.net.forward(img5).squeeze().cpu().data.numpy()#.squeeze(1)
        mask1 = mask[:4] + mask[4:,:,::-1]
        mask2 = mask1[:2] + mask1[2:,::-1]
        mask3 = mask2[0] + np.rot90(mask2[1])[::-1,::-1]
        
        return mask3

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

######################################################################################################

@njit        
def narrowize(grid,bound,thickness):
    """
    Narrows down the roads by the specified thickness
    """
    x,y = grid.shape
    ngrid = grid
    for i in range(x-1):
        for j in range(y-1):
            if bound[i,j]==1:
                ngrid[i-thickness:i+thickness,j] = 1
                ngrid[i,j-thickness:j+thickness] = 1
    return ngrid   

def split_stitch(image,masks=None):
    """
    Returns an array containing 1024x1024px slices of the image

    If masks = [img1,img2,...], 
    Returns the image stitched together from masks
    """
    h,w = image.shape[:2]
    out = []
    if w>h:        
        
        splits = np.floor(w/h).astype('int') 
        if masks == None:

            for i in range(splits):
                sq = cv2.resize(image[:,i*h:(i+1)*h],(1024,1024))

                out.append(sq)
            last = cv2.resize(image[:,-h:],(1024,1024))

            out.append(last)
            return out 
        else:

            for mask in masks:
                out.append(cv2.resize(mask,(h,h)))
            fabric = np.concatenate(out[:-1],axis=1) 
            last = out[-1][:,(h*splits)-w:] 
            fabric = np.concatenate((fabric,last),axis=1)
            
            return fabric
    elif w<h:
        
        splits = np.floor(h/w).astype('int')
        if masks == None:
            for i in range(splits):
                sq = cv2.resize(image[i*w:(i+1)*w,:],(1024,1024))
                out.append(sq)
            last = cv2.resize(image[-w:,:],(1024,1024))
            out.append(last)
            return out 
        else:
            for mask in masks:
                out.append(cv2.resize(mask,(w,w)))
            fabric = np.concatenate(out[:-1],axis=0)
            last = out[-1][(w*splits)-h:,:]
            fabric = np.concatenate((fabric,last),axis=0)    
            return fabric    
    else:
        if masks == None:
            out.append(cv2.resize(image,(1024,1024))) 
            return out  
        else:
            fabric = cv2.resize(masks[0],(h,h))
            return fabric


def euclidean(node1, node2):
    '''
    Returns the eulidean distance beween two node coordinates. 
    '''
    x1,y1 = node1
    x2,y2 = node2
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)

def initgraph(grid,draw=False):
    '''
    Initializes the possible nodes in a binary occupancy grid (2d numpy array).
    '''
    grid_size = grid.shape
    G = nx.grid_2d_graph(*grid_size)
    deleted_nodes = 0 
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i,j] == 1:
                G.remove_node((i,j))
                deleted_nodes += 1
          
    if draw:
        print(f"removed {deleted_nodes} nodes")
        print(f"number of occupied cells in grid {np.sum(grid)}")
        pos = {(x,y):(y,-x) for x,y in G.nodes()}
        nx.draw(G, pos = pos, node_color = 'red', node_size=2)
    return G

def generate_path(Graph,start,end,heur = euclidean):
    '''
    Generates the A* path using the given start and end nodes with the given heuristic.
    If node not found, finds the nearest (euclidean) node on the graph.
    '''
    startx,endx = (-1,-1),(-1,-1)
    if not Graph.has_node(tuple(start)):
        node_coords = np.array(nx.nodes(Graph))
        tree = KDTree(node_coords, metric='minkowski')
        start_idx = tree.query(np.array(start).reshape(1,-1), k=1, return_distance=False)[0]
        startx = tuple(node_coords[start_idx][0])
        print(startx)
    if not Graph.has_node(tuple(end)):
        node_coords = np.array(nx.nodes(Graph))
        tree = KDTree(node_coords, metric='minkowski')
        end_idx = tree.query(np.array(end).reshape(1,-1), k=1, return_distance=False)[0]
        endx = tuple(node_coords[end_idx][0] )
        print(endx)
    try:
        if startx[1] == -1 and endx[1] == -1:
            astar_path = nx.astar_path(Graph, start, end, heuristic=heur, weight="weight") 
        else:
            if startx[1] != -1:
                if endx[1] == -1:
                    astar_path = [start] + nx.astar_path(Graph, startx, end, heuristic=heur, weight="weight")
                else:
                    astar_path = [start] + nx.astar_path(Graph, startx, endx, heuristic=heur, weight="weight") + [end]
            elif endx[1] != -1:
                astar_path = nx.astar_path(Graph, start, endx, heuristic=heur, weight="weight") + [end]

    except nx.NetworkXNoPath:
        return None   

    
    return astar_path    

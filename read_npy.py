import numpy as np
import os
import math
import cv2 as cv
import random
from collections import OrderedDict
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
cmaps = OrderedDict()
# folder = 'debug'
# npy_list = os.listdir(folder)
# for npy_file in npy_list:
#     var = np.load(os.path.join(folder, npy_file))
#     if '.' in npy_file:
#         print('{}: {}'.format(npy_file, var))




anchor =  np.load('./priorbox_data.npy')
anchor = np.reshape(anchor, (-1,4))
print('anchor shape = {}'.format(anchor.shape))
print('anchor shape = {}'.format(anchor[:5,:]))


def gen_map(length = 36):
    length = length // 6 * 6
    h = range(length)
    s = np.ones((1, length // 6))
    v = np.ones((1, length // 6))
    f = np.array([(x / 60) - (x % 6) for x in h])
    f = np.reshape(f, [6, -1])
    p = np.zeros((1, length // 6))
    q = 1. - f
    t = f
    t = np.reshape(t, [6,-1])
    rgb = []
    print('s={},f={},p={},q={},t={}, v={}'.format(s.shape,f.shape,p.shape,q.shape,t.shape,v.shape))

    print('v={},t={},p={}'.format(v,t[0,:],p))
    # rgb1 = np.stack([v,[t[0,:]],p], axis=-1)
    rgb.append(np.stack([v,[t[0,:]],p], axis=-1))
    rgb.append(np.stack([[q[1,:]],v,p], axis=-1))
    rgb.append(np.stack([p,v,[t[2,:]]], axis=-1))
    rgb.append(np.stack([p,[q[3,:]],v], axis=-1))
    rgb.append(np.stack([[t[4,:]],p,v], axis=-1))
    rgb.append(np.stack([v,p,[q[5,:]]], axis=-1))
    rgb = np.reshape(np.stack(rgb, axis = 0), [length, 3])
    return rgb




def center2point( center_y, center_x, height, width):
    angle = math.atan((center_y - 0.5)/( center_x - 0.5 + 1e-20))
    pi = math.pi
    angle = np.where(np.less_equal(center_x, 0.5) , pi - angle, -angle)
    angle = np.where(np.less_equal(angle, 0.0), pi + pi + angle, angle)
    
    rotation_matrix = np.stack([math.cos(angle), -math.sin(angle),  
                        math.sin(angle),math.cos(angle)], axis=0)
    rotation_matrix = np.reshape(rotation_matrix, (2, 2))
    height, width = width, height 
    points = np.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
    points = np.matmul(points, rotation_matrix) + [center_x, center_y]
    return points


image = np.zeros((512,512,3))
# anchor = anchor[3:1444:4,:]
# anchor = anchor[3+1444:1844:4,:]
anchor = anchor[-40::4,:]
color = gen_map()
for i in range(anchor.shape[0]):
    yxhw = anchor[i,:]
    points = center2point( yxhw[0], yxhw[1], yxhw[2], yxhw[3])
    points = np.multiply(points, 512).astype(int)
    points = np.reshape(points.astype(int), [-1,1,2])
    cv.polylines(image,[points],True,color[(i*7)%36,:],1 + i%2) 
cv.imshow('anchor', image)
cv.waitKey()


import numpy as np
import cv2 as cv
import os
# seed the pseudorandom number generator
from random import seed
from random import randint


def plot_xy(x_data, y_data, name):
    x_data = np.reshape(x_data, [-1,1])
    y_data = np.reshape(y_data, [-1,1])
    if len(x_data) != len(y_data):
        return
    image = np.zeros((1000,1000,3))
    for x, y in zip(x_data, y_data):
        xx = int(x*400 + 500)
        yy = int(y*400 + 500)
        cv.circle(image, (xx, yy), 1, (randint(150,255),randint(100,255),randint(200,255)), 1)
    cv.imshow(name, image/255)
    # cv.waitKey()

def plot_y(y_data, name):
    
    w = 1000
    h = 500
    y_data = np.reshape(y_data, [-1,1])
    image = np.zeros((h,w,3))
    x_data =  np.reshape(range(0, len(y_data)), [-1,1])/w

    ymax = max(y_data)
    ymin = min(y_data)
    yrange = ymax - ymin
    print(ymax, ymin)

    x_data = (x_data * w).astype(int)
    y_data = ((y_data-ymin)/yrange * h).astype(int)
    if len(x_data) != len(y_data):
        return
    for i in range(len(x_data) - 1):
        cv.line(image, (x_data[i], y_data[i]), (x_data[i+1], y_data[i+1]), (255,155,200), 1)
    
    cv.imshow(name, image/255)
    # cv.waitKey()




for i in range(1,10):
    gt_target = np.loadtxt('./debug/gt_cx_{}.txt'.format(i))
    # print('conf = \n{}\n{}'.format(len(gt_target), gt_target))
    plot_y(gt_target[:,0], 'cy_{}'.format(i))
    plot_y(gt_target[:,1], 'cx_{}'.format(i))
    plot_y(gt_target[:,2], 'h_{}'.format(i))
    plot_y(gt_target[:,3], 'w_{}'.format(i))

cv.waitKey()

# plot_xy(tf_loc_data,loc, 'location')
# plot_xy(tf_priorbox,priorbox, 'anchor')
# plot_xy(tf_conf_data,conf, 'conf')

# cv.waitKey()
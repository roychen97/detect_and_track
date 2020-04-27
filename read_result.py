import numpy as np
import cv2 as cv
import os
# seed the pseudorandom number generator
from random import seed
from random import randint

# seed(1)
def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """
    orig_shape = x.shape

    if len(x.shape) > 1:
        # Matrix
        exp_minmax = lambda x: np.exp(x - np.max(x))
        denom = lambda x: 1.0 / np.sum(x)
        x = np.apply_along_axis(exp_minmax,1,x)
        denominator = np.apply_along_axis(denom,1,x) 
        
        if len(denominator.shape) == 1:
            denominator = denominator.reshape((denominator.shape[0],1))
        
        x = x * denominator
    else:
        # Vector
        x_max = np.max(x)
        x = x - x_max
        numerator = np.exp(x)
        denominator =  1.0 / np.sum(numerator)
        x = numerator.dot(denominator)
    
    assert x.shape == orig_shape
    return x
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




path = 'ssdface_freeze/debug4'

# ＝＝＝ tx2 tx2 tx2
priorbox = np.loadtxt(os.path.join(path,'loc_data.txt'))
# print('priorbox = \n{}'.format(priorbox))

conf = np.loadtxt(os.path.join(path,'conf_data.txt'))
# loc = np.reshape(loc, [-1,4])
# conf = softmax(np.reshape(conf, [-1,2]))
print('conf = \n{}'.format(conf))
img = np.transpose(np.reshape(conf, [3,300,300]),[1,2,0])
cv.imshow('input_img tx2', img + 1)

loc = np.loadtxt(os.path.join(path,'priorbox_data.txt'))
# print('loc = \n{}'.format(loc))

# ＝＝＝

# ＝＝＝ tensorflow
tf_priorbox  =  np.load(os.path.join(path,'tf_priorbox.npy'))
tf_conf_data  =  np.load(os.path.join(path,'tf_conf_data.npy'))
img = np.transpose(np.reshape(tf_conf_data, [3,300,300]),[1,2,0])
cv.imshow('input_img', img + 1)
tf_loc_data  =  np.load(os.path.join(path,'tf_loc_data.npy'))
print('tf_loc_data shape =\n{}'.format(tf_loc_data.shape))
print('tf_conf_data=\n{}'.format(tf_conf_data.shape))
# cv.waitKey()
# tf_conf_data = np.reshape(tf_conf_data,[-1,2])
# maps = []
# maps.append(tf_conf_data[0:19*19*6,1])
# maps.append(tf_conf_data[1444:2044,1])
# maps.append(tf_conf_data[2044:2194,1])
# maps.append(tf_conf_data[2194:2248,:])
# maps.append(tf_conf_data[2248:2264,:])
# maps.append(tf_conf_data[2264:,:])

# size = [19,10,5,3,2,1]

# for m, map in enumerate(maps):
#     if m < 2:
#         continue
#     feature = np.reshape(map,[size[m], size[m], -1])
#     print(feature)
#     for i in range(feature.shape[-1]):
#         cv.imshow('feature_{}_{}'.format(m,i), feature[:,:, i])
# cv.waitKey()



# ＝＝＝


plot_xy(tf_loc_data,loc, 'location')
plot_xy(tf_priorbox,priorbox, 'anchor')
plot_xy(tf_conf_data,conf, 'conf')

cv.waitKey()
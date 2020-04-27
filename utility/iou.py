import math

import tensorflow as tf
import cv2 as cv
import numpy as np

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"


    a = np.array([[[1,0,0], [1,0,0], [1,1,1]], [[0,0,0], [0,1,0], [1,0,0]],[[0,0,0], [1,1,1], [1,1,1]]])
    b = np.array([[[0,0,0], [0,0,1], [0,0,1]], [[0,0,1], [1,0,0], [0,1,0]]])
    tfa = tf.constant(a)
    tfb = tf.constant(b) 
    tfa = tf.reshape(tfa,[1,-1,3,3])
    tfb = tf.reshape(tfb,[-1,1,3,3])
    matrix = tf.multiply(tfa,tfb)
    sess = tf.Session()
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    m,aa,bb = sess.run([matrix,tfa,tfb])
    print('matirx = \n{}\n{}'.format(m.shape, m))
    print('aa = \n{}\n{}'.format(aa.shape, aa))
    print('bb = \n{}\n{}'.format(bb.shape, bb))
    input()
    def center2point( box):
        center_y, center_x, height, width = box[0], box[1], box[2], box[3]
        angle = tf.math.atan(tf.truediv(center_y - 0.5, center_x - 0.5))
        pi = tf.constant(math.pi)
        # angle = tf.cond(tf.less_equal(center_x, tf.constant(0.5)), lambda: tf.subtract(pi,  angle), lambda: tf.subtract(tf.constant(0.0),  angle))
        # angle = tf.cond(tf.less_equal(angle, tf.constant(0.0)), lambda: tf.add(pi + pi,  angle), lambda: tf.identity(angle))
        angle = tf.where(tf.less_equal(center_x, tf.constant(0.5)) , pi - angle, -angle)
        angle = tf.where(tf.less_equal(angle, tf.constant(0.0)), pi + pi + angle, angle)
        rotation_matrix = tf.stack([tf.cos(angle), -tf.sin(angle),  
                            tf.sin(angle),tf.cos(angle)], axis=0)
        rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
        points = tf.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
        points = tf.matmul(points, rotation_matrix) + [center_x, center_y]
        return points, angle

    def fill_poly(points, size):
        img = np.zeros((size,size))
        points = (points * size).astype(int)
        # points = np.stack([[int(pt[0] * size), int(pt[1] * size)] for pt in points], axis = 0)
        cv.fillPoly(img, [points], 255)
        return img 

    sess = tf.Session()
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    
    def run_pt(x,y,w,h,name):
        center_x = tf.constant(x)
        center_y = tf.constant(y)
        width = tf.constant(w)
        height = tf.constant(h)
        center_y, center_x, height, width = [tf.reshape(b, [-1]) for b in [center_y, center_x, height, width]]
        
        # # # run map function: center size to 4 corner points
        input_value = tf.stack([center_y, center_x, height, width], axis = 1)
        points, angle = tf.map_fn(center2point, input_value, (tf.float32, tf.float32))

        func = lambda points: tf.py_func(fill_poly, [points, 512], tf.float64)
        # points = tf.constant([[[0.1,0.1],[0.6,0.3],[0.9,0.6],[0.2,0.7]],
        #                     [[0.1,0.8],[0.3,0.5],[0.7,0.5],[0.1,0.1]],
        #                     [[0.9,0.3],[0.1,0.6],[0.4,0.4],[0.5,0.2]]], tf.float64)
        image = tf.map_fn(func, tf.cast(points, tf.float64))
        image = sess.run(image)
        image.astype(np.uint8)
        for i in range(image.shape[0]):
            cv.imshow(name + str(i), image[i])

        size = 512
        points = tf.cast(tf.multiply(points, tf.constant(512.0)), tf.int32)
        points, angle = sess.run([points, angle])        
        img = np.zeros((size, size,3), dtype=np.uint8)
        for i in range(len(points)):
            pts = points[i].reshape((-1,1,2))
            print('angle = {}, pts = {}, {}'.format(int(angle[i]*180/3.1415926),pts.shape, pts))
            cv.polylines(img,[pts],True,(0,255,255))
        cv.imshow(name, img)
        cv.waitKey()




    run_pt([0.7, 0.3, 0.55],[0.6,0.1,0.4],[0.08,0.3,0.07],[0.02,0.1,0.05],'1')
    
    # size = 16
    # image = get_box_mask(0.3,0.6,0.1,0.2)
    # image = sess.run([image])
    # image = np.array(image, np.uint8)
    # print(image[0].shape)
    # cv.imshow('mask', image[0])

    cv.waitKey()
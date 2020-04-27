# Copyright 2018 Changan Wang

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import math
import tensorflow as tf
# from scipy.misc import imread, imsave, imshow, imresize
import numpy as np
from tensorflow.python.tools import freeze_graph
#from net import ssd_net#_48 as ssd_net

from dataset import dataset_common
from preprocessing import ssd_preprocessing
from utility import anchor_manipulator
from utility import draw_toolbox
import time
import cv2 as cv
from make_npy import make_npy
from easydict import EasyDict as edict
import json


# config_name = 'config/mobilenetv2_300_person_wework.json'
# config_name = 'config/mobilenetv2_300_person_wework_anchor2.json'
config_name = 'config/mobilenetv2_300_person_wework_480.json'

# config_name = 'config/mobilenetv2_300_person.json'
# config_name = 'config/mobilenetv2_96_face.json'

sys.path.insert(0, 'net')
with open(config_name, 'r') as config_file:
    config_args_dict = json.load(config_file)
    config_args = edict(config_args_dict)
ssd_net = __import__(config_args.model_file)



def layer_params(layers_shapes, anchor_scales, anchor_ratios, extra_anchor_scales):
    feature_num = []
    feature_depth = []
    anchor_num = 0
    for id, layer in enumerate(layers_shapes):
        feature_num.append(layer[0] * layer[1])
        feature_depth.append(len(anchor_scales[id]) *len(anchor_ratios[id])  + len(extra_anchor_scales[id]))
        anchor_num += (layer[0] * layer[1])* feature_depth[-1]

    return anchor_num, feature_num, feature_depth
    

# scaffold related configuration
tf.app.flags.DEFINE_integer(
    'num_classes', 2, 'Number of classes to use in the dataset.')
# model related configuration
tf.app.flags.DEFINE_integer(
    'train_image_size', config_args.input_size,
    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(
    'data_format', 'channels_first', # 'channels_first' or 'channels_last'
    'A flag to override the data format used in the model. channels_first '
    'provides a performance boost on GPU but is not always compatible '
    'with CPU. If left unspecified, the data format will be chosen '
    'automatically based on whether TensorFlow was built for CPU or GPU.')
tf.app.flags.DEFINE_float(
    'select_threshold', 0.2, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(
    'min_size', 0.1, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(
    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(
    'nms_topk', 20, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(
    'keep_topk', 100, 'Number of total object to keep for each image before nms.')
# checkpoint related configuration
tf.app.flags.DEFINE_string(
    'checkpoint_path', config_args.logdir,
    'The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string(
    'model_scope', 'ssd300',
    'Model scope name used to replace the name_scope in checkpoint.')

FLAGS = tf.app.flags.FLAGS
#CUDA_VISIBLE_DEVICES

def get_checkpoint():
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    return checkpoint_path

def select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold):
    selected_bboxes = {}
    selected_scores = {}
    with tf.name_scope('select_bboxes', values=[scores_pred, bboxes_pred]):
        for class_ind in range(1, num_classes):
            class_scores = scores_pred[:, class_ind]
            # select_mask = class_scores > select_threshold
            # select_mask = tf.cast(select_mask, tf.float32)
            select_mask = tf.where(class_scores > select_threshold, tf.ones_like(class_scores), tf.zeros_like(class_scores))
            selected_bboxes[class_ind] = tf.multiply(bboxes_pred, tf.expand_dims(select_mask, axis=-1))
            selected_scores[class_ind] = tf.multiply(class_scores, select_mask)
        print('\n==\nselected_bboxes = {}'.format(selected_bboxes))
        print('\n==\nselected_scores = {}'.format(selected_scores))
    return selected_bboxes, selected_scores

def clip_bboxes(cy, cx, h, w, name):
    with tf.name_scope(name, 'clip_bboxes', [cy, cx, h, w]):
        cy = tf.maximum(cy, 0.)
        cx = tf.maximum(cx, 0.)
        cy = tf.minimum(cy, 1.)
        cx = tf.minimum(cx, 1.)
        h = tf.maximum(h, 0.)
        w = tf.maximum(w, 0.)
        h = tf.minimum(h, 1.)
        w = tf.minimum(w, 1.)

        return cy, cx, h, w

def filter_bboxes(scores_pred, ymin, xmin, ymax, xmax, min_size, name):
    with tf.name_scope(name, 'filter_bboxes', [scores_pred, ymin, xmin, ymax, xmax]):
        width = xmax - xmin
        height = ymax - ymin

        filter_mask = tf.logical_and(width > min_size, height > min_size)

        filter_mask = tf.cast(filter_mask, tf.float32)
        return tf.multiply(ymin, filter_mask), tf.multiply(xmin, filter_mask), \
                tf.multiply(ymax, filter_mask), tf.multiply(xmax, filter_mask), tf.multiply(scores_pred, filter_mask)

def sort_bboxes(scores_pred, cy, cx, h, w, keep_topk, name):
    with tf.name_scope(name, 'sort_bboxes', [scores_pred, cy, cx, h, w]):
        cur_bboxes = tf.shape(scores_pred)[0]
        scores, idxes = tf.nn.top_k(scores_pred, k=tf.minimum(keep_topk, cur_bboxes), sorted=True)

        cy, cx, h, w = tf.gather(cy, idxes), tf.gather(cx, idxes), tf.gather(h, idxes), tf.gather(w, idxes)

        paddings_scores = tf.expand_dims(tf.stack([0, tf.maximum(keep_topk-cur_bboxes, 0)], axis=0), axis=0)

        return tf.pad(cy, paddings_scores, "CONSTANT"), tf.pad(cx, paddings_scores, "CONSTANT"),\
                tf.pad(h, paddings_scores, "CONSTANT"), tf.pad(w, paddings_scores, "CONSTANT"),\
                tf.pad(scores, paddings_scores, "CONSTANT")

def nms_bboxes(scores_pred, bboxes_pred, nms_topk, nms_threshold, name):
    with tf.name_scope(name, 'nms_bboxes', [scores_pred, bboxes_pred]):
        idxes = tf.image.non_max_suppression(bboxes_pred, scores_pred, nms_topk, nms_threshold)
        return tf.gather(scores_pred, idxes), tf.gather(bboxes_pred, idxes)

def center2point( center_y, center_x, height, width):
    # print(center_y.shape)
    angle = np.arctan((center_y - 0.5)/( center_x - 0.5 + 1e-20))
    pi = math.pi
    angle = np.where(np.less_equal(center_x, 0.5) , pi - angle, -angle)
    angle = np.where(np.less_equal(angle, 0.0), pi + pi + angle, angle)
    # print(angle.shape)
    rotation_matrix = np.stack([math.cos(angle), -math.sin(angle),  
                        math.sin(angle),math.cos(angle)], axis=0)
    rotation_matrix = np.reshape(rotation_matrix, (2, 2))
    height, width = width, height 
    points = np.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
    points = np.matmul(points, rotation_matrix) + [center_x, center_y]
    return points

def fill_poly(points, size):
    img = np.zeros((int(size),int(size)))
    points = np.int32(points * size)
    # points = np.stack([[int(pt[0] * size), int(pt[1] * size)] for pt in points], axis = 0)
    cv.fillPoly(img, [points], 1)
    return img

def get_box_mask( center_y, center_x, height, width):
    center_y, center_x, height, width = [np.reshape(b, [-1]) for b in [center_y, center_x, height, width]]
    input_value = np.stack([center_y, center_x, height, width], axis = 1)
    points = [center2point(cy, cx, h, w ) for cy, cx, h, w in zip(center_y, center_x, height, width)]
    image = [fill_poly(pts, 96.) for pts in points]
    image = np.stack(image, axis=0)
    return image

# def compute_iou(mask1, mask2):
#     intersection = np.sum(mask1*mask2)
#     if intersection > 0.0:
#         return intersection / (sum(mask1) + sum(mask2) - intersection)
#     else:
#         return 0.0

def compute_iou(box, boxes, box_area, boxes_area):
    dist = np.reshape(np.square(box[0] - boxes[:,0] ) + np.square(box[1] - boxes[:,1]), [-1,1])
    size = np.reshape(np.square(np.maximum(np.maximum(box[2], box[3]), np.maximum(boxes[:,2], boxes[:,3]))), [-1,1])
    box_area = np.expand_dims(np.expand_dims(box_area, 0), 0)
    boxes_area = np.expand_dims(boxes_area, 1)
    cond = dist < size
    inter_area = np.sum(np.multiply(box_area, boxes_area), (2,3))
    # print('inter_area = {}, cond = {}'.format(inter_area.shape, cond.shape))
    # input()
    inter_vol = np.where(dist< size, np.sum(np.multiply(box_area, boxes_area), (2,3)), 0)
    area1 = np.sum(box_area, (2,3))
    area2 = np.sum(boxes_area, (2,3))
    union_vol = area1 + area2 - inter_vol
    return np.where(np.equal(union_vol, 0.0), np.zeros_like(inter_vol), inter_vol/ union_vol)



def non_max_suppression(scores_pred, bboxes_pred, nms_topk, nms_threshold):
    assert bboxes_pred.shape[0] == scores_pred.shape[0]
    cy = bboxes_pred[:, 0]
    cx = bboxes_pred[:, 1]
    h = bboxes_pred[:, 2]
    w = bboxes_pred[:, 3]
    masks = get_box_mask(cy, cx, h, w)
    # # box coordinate ranges are inclusive-inclusive
    # areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores_pred.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(bboxes_pred[index], bboxes_pred[scores_indexes], masks[index],
                           masks[scores_indexes])
        filtered_indexes = set((ious > nms_threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    
    if len(boxes_keep_index) > nms_topk:
      boxes_keep_index = boxes_keep_index[:nms_topk]
    return np.array(boxes_keep_index)




def parse_by_class(cls_pred, bboxes_pred, num_classes, select_threshold, min_size, keep_topk, nms_topk, nms_threshold):
    with tf.name_scope('select_bboxes', values=[cls_pred, bboxes_pred]):
        scores_pred = tf.nn.softmax(cls_pred)
        selected_bboxes, selected_scores = select_bboxes(scores_pred, bboxes_pred, num_classes, select_threshold)

        for class_ind in range(1, num_classes):
            cy, cx, h, w = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            ymin, xmin, ymax, xmax = tf.unstack(selected_bboxes[class_ind], 4, axis=-1)
            #ymin, xmin, ymax, xmax = tf.squeeze(ymin), tf.squeeze(xmin), tf.squeeze(ymax), tf.squeeze(xmax)
            cy, cx, h, w = clip_bboxes(cy, cx, h, w, 'clip_bboxes_{}'.format(class_ind))
            # ymin, xmin, ymax, xmax, selected_scores[class_ind] = filter_bboxes(selected_scores[class_ind],
            #                                     ymin, xmin, ymax, xmax, min_size, 'filter_bboxes_{}'.format(class_ind))
            cy, cx, h, w , selected_scores[class_ind] = sort_bboxes(selected_scores[class_ind],
                                                cy, cx, h, w , keep_topk, 'sort_bboxes_{}'.format(class_ind))
            selected_bboxes[class_ind] = tf.stack([cy, cx, h, w ], axis=-1)
            # selected_scores[class_ind], selected_bboxes[class_ind] = nms_bboxes(selected_scores[class_ind], selected_bboxes[class_ind], nms_topk, nms_threshold, 'nms_bboxes_{}'.format(class_ind))

        return selected_bboxes, selected_scores

def preprocessing(np_image, input_format, output_format, data_format, bias):
    # _R_MEAN = 123.68
    # _G_MEAN = 116.78
    # _B_MEAN = 103.94
    image = cv.resize(np_image, (config_args.input_size, config_args.input_size))
    image = image.astype(float)
    
    if input_format == 'bgr':
      test_image = np.concatenate([np.expand_dims(image[:,:,2], axis = -1),
                                  np.expand_dims(image[:,:,1], axis = -1), 
                                  np.expand_dims(image[:,:,0], axis = -1)], axis = 2)
    else:
      test_image = image                                          
    test_image = test_image - np.expand_dims(np.expand_dims(bias, axis = 0), axis = 0)
    if output_format == 'bgr':
      output_image = np.concatenate([np.expand_dims(test_image[:,:,2], axis = -1),
                                    np.expand_dims(test_image[:,:,1], axis = -1), 
                                    np.expand_dims(test_image[:,:,0], axis = -1)], axis = 2)
    else:
      output_image = test_image  
    if data_format == 'channels_first':
        output_image = np.transpose(output_image, (2, 0, 1))
    output_image = output_image * 2.0 / 255.0
    return np.expand_dims(output_image, axis = 0)


def main(_):
    make_npy('./priorbox_data.npy', config_args)
    with tf.Graph().as_default():
        out_shape = [FLAGS.train_image_size] * 2

        ''' 
        # tf not do preprocessing 
        image_input = tf.placeholder(tf.uint8, shape=(None, None, 3), name = 'input_image')
        shape_input = tf.shape(image_input)[:-1]
        #shape_input = tf.placeholder(tf.int32, shape=(2,), name = 'input_shape')
        features = ssd_preprocessing.preprocess_for_eval(image_input, out_shape, data_format=FLAGS.data_format, output_rgb=False)
        features = tf.expand_dims(features, axis=0)
        '''
        if FLAGS.data_format == 'channels_first':
            features = tf.placeholder(tf.float32, shape=(1, 3, config_args.input_size, config_args.input_size), name = 'input_image')
        else:
            features = tf.placeholder(tf.float32, shape=(1, config_args.input_size, config_args.input_size, 3), name = 'input_image')

        anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
            layers_shapes = config_args.layers_shapes,
            anchor_scales = config_args.anchor_scales,
            extra_anchor_scales = config_args.extra_anchor_scales,
            anchor_ratios = config_args.anchor_ratios,
            layer_steps = config_args.layer_steps)

        all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
        
        anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders = [1.0] * config_args.layer_num,
                                                        positive_threshold = None,
                                                        ignore_threshold = None,
                                                        prior_scaling=[0.1, 0.1, 0.1, 0.1])
        decode_fn = lambda pred : anchor_encoder_decoder.ext_decode_all_anchors(pred, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)

        with tf.variable_scope(FLAGS.model_scope, default_name=None, values=[features], reuse=tf.AUTO_REUSE):
            backbone = ssd_net.MobileNetV2(FLAGS.data_format)
            feature_layers = backbone.forward(features, training=False)
            location_pred, cls_pred = ssd_net.multibox_head(feature_layers, FLAGS.num_classes, all_num_anchors_depth, data_format=FLAGS.data_format)
            if FLAGS.data_format == 'channels_first':
                cls_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in cls_pred]
                location_pred = [tf.transpose(pred, [0, 2, 3, 1]) for pred in location_pred]
                        
            anchor_num, feature_num, feature_depth = layer_params(config_args.layers_shapes, config_args.anchor_scales, config_args.anchor_ratios, config_args.extra_anchor_scales)

            cls_pred = [tf.reshape(pred, [1, -1, FLAGS.num_classes, 1])  for pred in cls_pred]
            location_pred = [tf.reshape(pred, [1, -1, 4, 1]) for pred in location_pred]            
            cls_pred = tf.concat(cls_pred, axis=1, name = 'concat_cls')
            location_pred = tf.concat(location_pred, axis=1, name = 'concat_loc')


        with tf.device('/cpu:0'):
            location_pred_pos = tf.reshape(location_pred, tf.shape(location_pred)[1:-1])
            cls_pred_pos = tf.reshape(cls_pred, tf.shape(cls_pred)[1:-1])
            bboxes_pred = decode_fn(location_pred_pos)
            bboxes_pred = tf.concat(bboxes_pred, axis=0)
            selected_bboxes, selected_scores = parse_by_class(cls_pred_pos, bboxes_pred,
                                                            FLAGS.num_classes, FLAGS.select_threshold, FLAGS.min_size,
                                                            FLAGS.keep_topk, FLAGS.nms_topk, FLAGS.nms_threshold)
            labels_list = []
            scores_list = []
            bboxes_list = []
            for k, v in selected_scores.items():
                labels_list.append(tf.ones_like(v, tf.int32) * k)
                scores_list.append(v)
                bboxes_list.append(selected_bboxes[k])
            all_labels = tf.concat(labels_list, axis=0)
            all_scores = tf.concat(scores_list, axis=0)
            all_bboxes = tf.concat(bboxes_list, axis=0)
            all_pres = tf.concat([tf.expand_dims(all_scores, axis = 1), all_bboxes], axis=1)

        
        saver = tf.train.Saver()
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            checkpoint_path = get_checkpoint()
            saver.restore(sess, checkpoint_path)
            tf.summary.FileWriter('./freezelogs', sess.graph)

            ''' 
            # show nodmes
            node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            for name in node_names:
                print(name)
            '''

            # freeze model exclude preprocessing and postporcessing
            priorbox_np = np.reshape(np.load('./priorbox_data.npy'), [-1,1,anchor_num*4,1])
            scale_np = np.reshape(np.matmul(np.ones((anchor_num,1)), [[0.1, 0.1, 0.1, 0.1]]), [-1,1,anchor_num*4,1])
            priorbox_data = tf.concat([tf.constant(priorbox_np, dtype=tf.float32), tf.constant(scale_np, dtype=tf.float32)], axis = 1)
            priorbox_data = tf.reshape(priorbox_data, [-1, 2 , anchor_num*4,1], name = 'priorbox_data' )
            # cls_pred = tf.nn.softmax(cls_pred, axis=2)
            loc_data = tf.reshape(location_pred, [-1, anchor_num*4, 1, 1], name = 'loc_data')
            conf_data = tf.reshape(cls_pred, [-1, anchor_num*2, 1, 1], name = 'conf_data')
            tf.train.write_graph(sess.graph_def, 'model', 'ssd_model.pb')
            freeze_graph.freeze_graph('model/ssd_model.pb',
                      '', False, checkpoint_path  , 'loc_data,conf_data, priorbox_data', 'save/restore_all', 'save/Const:0',
                       'ssdface_freeze/' + config_args.freeze_file + '.pb', False, "")
            print("FREEZE done")
 

            # freeze model exclude preprocessing 
            # outputs = tf.identity(all_pres, name='outputs')
            # tf.train.write_graph(sess.graph_def, 'model', 'face_ssd_model_nonpreprocess.pb')
            # freeze_graph.freeze_graph('model/face_ssd_model_nonpreprocess.pb',
            #           '', False, checkpoint_path  , 'outputs', 'save/restore_all', 'save/Const:0',
            #            'ssdface_freeze/face_ssd_model_fz.pb', False, "")
            # print("FREEZE done")


            def read_labelme(json_path):
                if not os.path.isfile(json_path) :
                    return []
                if not '.json' in json_path:
                    return False
                with open(json_path) as json_file:
                    data = json.load(json_file)
                all_boxes = []
                if 'shapes' in data:
                    for obj in data['shapes']:
                        box = obj['points']
                        if len(box):
                            box = [[box[0][0], box[0][1]], [box[1][0],box[1][1]], [box[2][0],box[2][1]], [box[3][0],box[3][1]]]
                        # box = (map(int, box))
                        all_boxes.append(box)
                return all_boxes


            def test_img(filename):
                np_image = cv.imread(filename)
                # np_image = np_image[:,:np_image.shape[1],:]
                # np_image = np.ones((1,3,96,96), dtype=np.float32) * 0.5
                show_image = np_image.copy()
                np_image = preprocessing(np_image, input_format = 'bgr', output_format = 'rgb', data_format = FLAGS.data_format, bias = config_args.bias)

                start = time.time()
                # all_pres_, feature_layers_, all_anchors_ = sess.run(
                #     [all_pres, feature_layers, all_anchors], 
                #     feed_dict = {features : np_image}) 
                labels_, scores_, bboxes_ = sess.run(
                    [all_labels, all_scores, all_bboxes], 
                    feed_dict = {features : np_image})
                idxes = non_max_suppression(scores_, bboxes_, 20, 0.45)
                scores = np.take(scores_, idxes)
                bboxes = np.take(bboxes_, idxes, axis = 0)
                labels = np.take(labels_, idxes)


                end = time.time()
                img_to_draw = draw_toolbox.rotate_bboxes_draw_on_img(show_image, labels, scores, bboxes, thickness=5, xy_order = 'xy', color=(155,155,255))
                # img_to_draw = draw_toolbox.bboxes_draw_on_img(show_image, labels_, scores_, bboxes_, thickness=5, xy_order='yx')

                polygon = read_labelme(filename[:-4] + '.json')
                # print(polygon)
                for poly in polygon:
                    poly = np.stack(poly, axis=0).astype(int)
                    cv.polylines(img_to_draw,[poly.astype(int)],True,(0,255,255),2)

                # cv.putText(img_to_draw, "tracker = {:.0f} ms".format((end - start)/10 * 1000), (30,60), 2, 0.7, (255,55,155))
                cv.imwrite('./demo/' + filename.split('/')[-1], img_to_draw)
                # cv.imshow('./demo/test_out.jpg', img_to_draw[:,:,:])
                # print('\nrun time = {:.0f}ms'.format((end - start)/10 * 1000))
                

                cv.waitKey()

            # test image
            # test_folder = '/home/scchiu/Data/HABBOF/Meeting1' 
            # file_list = os.listdir(test_folder)
            # for filename in file_list:
            #     if 'jpg' in filename:
            #         test_img(os.path.join(test_folder, filename))
            # test_img('./demo/test.jpg')
            


            # test video/webcam



            cnt = 0
            time2 = 0
            start = 0
            cap = cv.VideoCapture('/home/scchiu/Data/WEWORK/test_images/video010.mp4')

            fourcc = cv.VideoWriter_fourcc(*'XVID')
            out = cv.VideoWriter('./video010' + '_output.mp4', fourcc, 15.0, (1024, 1024))

            a = True
            while(a):
                cnt += 1
                
                ret, frame = cap.read() 

                labels_, scores_, bboxes_ = sess.run( [all_labels, all_scores, all_bboxes], 
                    feed_dict = {features : preprocessing(frame, input_format = 'bgr', output_format = 'rgb', data_format = FLAGS.data_format, bias = config_args.bias)})
                idxes = non_max_suppression(scores_, bboxes_, 20, 0.45)
                scores = np.take(scores_, idxes)
                bboxes = np.take(bboxes_, idxes, axis = 0)
                labels = np.take(labels_, idxes)


                end = time.time()
                img_to_draw = cv.resize(frame, (600,600))
                img_to_draw = draw_toolbox.rotate_bboxes_draw_on_img(img_to_draw, labels, scores, bboxes, thickness=2, xy_order = 'xy', color=(155,155,255))


                if cnt > 9:
                    time2 = (time.time() - start ) / 10.
                    start = time.time()
                    cnt = 0
                cv.putText(frame, "tracker = {:.0f} ms".format(time2*1000), (30,60), 2, 0.7, (255,55,155))

                cv.imshow("webCam", img_to_draw)
                out.write(cv.resize(img_to_draw, (1024,1024)))
                #if a == False:
                #    cv.waitKey(0)
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break                
            out.release()

if __name__ == '__main__':

  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()

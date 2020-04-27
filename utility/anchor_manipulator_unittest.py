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

import tensorflow as tf
# from scipy.misc import imsave, imshow, imresize
import numpy as np
import sys; sys.path.insert(0, ".")
from utility import draw_toolbox
from utility import anchor_manipulator
from preprocessing import ssd_preprocessing
import cv2 as cv
slim = tf.contrib.slim

def save_image_with_bbox(image, labels_, scores_, bboxes_, gt_labels, gt_bboxes):
    if not hasattr(save_image_with_bbox, "counter"):
        save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
    save_image_with_bbox.counter += 1

    img_to_draw = np.copy(image)
    img_to_draw = draw_toolbox.rotate_bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=8)
    img_to_draw = draw_toolbox.rotate_bboxes_draw_on_img(img_to_draw, gt_labels, scores_, gt_bboxes, thickness=3, color=(255,128,200))
    # img_to_draw = draw_toolbox.bboxes_draw_on_img(img_to_draw, labels_, scores_, bboxes_, thickness=2)
    cv.imwrite(os.path.join('./debug/{}.jpg').format(save_image_with_bbox.counter), cv.cvtColor(img_to_draw, cv.COLOR_BGR2RGB) )
    return save_image_with_bbox.counter

def slim_get_split(file_pattern='{}_????', anchor_area=None):
    # Features in Pascal VOC TFRecords.
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/height': tf.FixedLenFeature([1], tf.int64),
        'image/width': tf.FixedLenFeature([1], tf.int64),
        'image/channels': tf.FixedLenFeature([1], tf.int64),
        'image/shape': tf.FixedLenFeature([3], tf.int64),
        'image/object/bbox/cx': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/cy': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/width': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/height': tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        'image/object/bbox/truncated': tf.VarLenFeature(dtype=tf.int64),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'shape': slim.tfexample_decoder.Tensor('image/shape'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
                ['cy', 'cx',  'height', 'width'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
        'object/difficult': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        'object/truncated': slim.tfexample_decoder.Tensor('image/object/bbox/truncated'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
                data_sources=file_pattern,
                reader=tf.TFRecordReader,
                decoder=decoder,
                num_samples=100,
                items_to_descriptions=None,
                num_classes=21,
                labels_to_names=None)

    with tf.name_scope('dataset_data_provider'):
        provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    num_readers=2,
                    common_queue_capacity=32,
                    common_queue_min=8,
                    shuffle=True,
                    num_epochs=1)

    [org_image, shape, glabels_raw, gbboxes_raw, isdifficult] = provider.get(['image', 'shape',
                                                                         'object/label',
                                                                         'object/bbox',
                                                                         'object/difficult'])

    image, glabels, gbboxes = ssd_preprocessing.preprocess_image(org_image, glabels_raw, gbboxes_raw, [128.,128.,128.], [300, 300], is_training=True, data_format='channels_last', output_rgb=True)
    # save_image_op1 = tf.py_func(save_image_with_bbox,
    #                         [ssd_preprocessing.unwhiten_image(image, [128.,128.,128.]),
    #                         tf.clip_by_value(tf.concat(glabels, axis=0), 0, tf.int64.max),
    #                         tf.concat(gbboxes, axis=0),
    #                         tf.concat(gbboxes, axis=0),
    #                         tf.clip_by_value(tf.concat(glabels, axis=0), 0, tf.int64.max),
    #                         tf.concat(gbboxes, axis=0)],
    #                         tf.int64, stateful=True)

    anchor_creator = anchor_manipulator.AnchorCreator([300] * 2,
            layers_shapes = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)],
            anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
            extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
            anchor_ratios = [(1., .5,), (1., .5, 0.3333), (1., .5, 0.3333), (1., .5, 0.3333), (1., .5,), (1., .5,)],
            layer_steps = [16, 32, 64, 100, 150, 300])    
    # anchor_creator = anchor_manipulator.AnchorCreator([300] * 2,
    #         layers_shapes = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    #         anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
    #         extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
    #         anchor_ratios = [(2., .5), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)],
    #         layer_steps = [8, 16, 32, 64, 100, 300])

    '''anchor_creator = anchor_manipulator.AnchorCreator([48] * 2,
            layers_shapes = [ (16, 16), (12, 12), (8,8), (4,4)],
            anchor_scales = [ (0.05,), (0.1,), (0.2,), (0.3,)],
            extra_anchor_scales = [ (0.08,),(0.15,), (0.25,), (0.4,)],
            anchor_ratios = [ (0.5,),(0.6,),(0.6,),(0.6,)],
            layer_steps = [ 3, 4, 6, 12])'''
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

    num_anchors_per_layer = []
    for ind in range(len(all_anchors)):
        num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
    anchor_area = np.load('./config/anchor_area.npy')
    anchor_encoder_decoder = anchor_manipulator.AnchorEncoder(allowed_borders=[1.0] * 6,
                                                        positive_threshold = 0.5,
                                                        ignore_threshold = 0.5,
                                                        prior_scaling=[0.1, 0.1, 0.2, 0.2],
                                                        anchor_area=anchor_area)
    # mask, points, box = anchor_encoder_decoder.calc_mask(glabels, gbboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, True)
    # return save_image_op1, mask, points, box
    # ioumatrix = anchor_encoder_decoder.encode_all_anchors(glabels, gbboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, True)
    # return ioumatrix

    gt_targets, gt_labels, gt_scores = anchor_encoder_decoder.encode_all_anchors(glabels, gbboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, True)
    anchors = anchor_encoder_decoder._all_anchors

    # # anchor_encoder_decoder.ext_decode_all_anchors(gt_targets, all_anchors, all_num_anchors_depth, all_num_anchors_spatial)

    # # split by layers
    gt_targets, gt_labels, gt_scores, anchors = tf.split(gt_targets, num_anchors_per_layer, axis=0),\
                                                tf.split(gt_labels, num_anchors_per_layer, axis=0),\
                                                tf.split(gt_scores, num_anchors_per_layer, axis=0),\
                                                [tf.split(anchor, num_anchors_per_layer, axis=0) for anchor in anchors]

    save_image_op = tf.py_func(save_image_with_bbox,
                            [ssd_preprocessing.unwhiten_image(image, [128.,128.,128.]),
                            tf.clip_by_value(tf.concat(gt_labels, axis=0), 0, tf.int64.max),
                            tf.concat(gt_scores, axis=0),
                            tf.concat(gt_targets, axis=0),
                            tf.clip_by_value(tf.concat(glabels, axis=0), 0, tf.int64.max),
                            tf.concat(gbboxes, axis=0)],
                            tf.int64, stateful=True)
    return save_image_op
if __name__ == '__main__':
#   os.environ['CUDA_VISIBLE_DEVICES'] = '1'
#   with tf.device('/gpu:1'):
    # save_image_op, overlap_matrix, points, box = slim_get_split('./dataset/wework_tfrecords/train*')
    anchor_area = np.load('./config/anchor_area.npy')
    save_image_op = slim_get_split('./dataset/wework/train-wework-*')
    # Create the graph, etc.
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])

    '''anchor_creator = anchor_manipulator.AnchorCreator([300] * 2,
            layers_shapes = [(30, 30), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
            anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)],
            extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)],
            anchor_ratios = [(2., ), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., 3., .5, 0.3333), (2., .5), (2., .5)],
            layer_steps = [10, 16, 32, 64, 100, 300])'''
    anchor_creator = anchor_manipulator.AnchorCreator([48] * 2,
            layers_shapes = [ (12, 12), (6, 6), (3, 3), (1, 1)],
            anchor_scales = [ (0.1,), (0.3,), (0.5,), (0.7,)],
            extra_anchor_scales = [ (0.2,),(0.4,), (0.6,), (0.8,)],
            anchor_ratios = [ (1.5,),(1.5,),(1.5,),(1.,)],
            layer_steps = [ 4, 8, 16, 48])
    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()

    # Create a session for running operations in the Graph.
    sess = tf.Session()
    # Initialize the variables (like the epoch counter).
    sess.run(init_op)

    # Start input enqueue threads.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    all_qanchors_ = sess.run(all_anchors)
    for layer, anchors in enumerate(all_qanchors_):
        print('\nlayer = {}, size = {}'.format(layer, len(anchors)))
        for n, anchor in enumerate(anchors):
            print('anchor {} = {}\n{}\n'.format(n, anchor.shape, anchor))
            if layer > 1:
                print(anchor)

    try:
        while not coord.should_stop():
            # Run training steps or whatever
            # save_image_op = sess.run(save_image_op)
            print(sess.run(save_image_op))
            # save_image_op, overlap_matrix, points, box = sess.run([save_image_op, overlap_matrix, points, box])
            
            # print('mask shape = {}, \nbox\n{}'.format(save_image_op.shape, save_image_op[-10:,:]))
            # cv.imshow('overlap_matrix',save_image_op/25)
            # input()
            # cv.waitKey()
            # print('cy = \n{}\n\n'.format(anchors_point[:,0]))
            # print('cx = \n{}\n\n'.format(anchors_point[:,1]))
            # print('h = \n{}\n\n'.format(anchors_point[:,2]))
            # print('w = \n{}\n\n'.format(anchors_point[:,3]))

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()

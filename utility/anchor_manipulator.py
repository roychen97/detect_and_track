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
import math

import tensorflow as tf
import cv2 as cv
import numpy as np

# from tensorflow.contrib.image.python.ops import image_ops

def areas(gt_bboxes):
    with tf.name_scope('bboxes_areas', values=[gt_bboxes]):
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        return (xmax - xmin) * (ymax - ymin)

def center2point( box):
    center_y, center_x, height, width = box[0], box[1], box[2], box[3]
    angle = tf.math.atan(tf.truediv(center_y - 0.5, center_x - 0.5))
    pi = tf.constant(math.pi)
    angle = tf.where(tf.less_equal(center_x, tf.constant(0.5)) , pi - angle, -angle)
    angle = tf.where(tf.less_equal(angle, tf.constant(0.0)), pi + pi + angle, angle)
    rotation_matrix = tf.stack([tf.cos(angle), -tf.sin(angle),  
                        tf.sin(angle),tf.cos(angle)], axis=0)
    rotation_matrix = tf.reshape(rotation_matrix, (2, 2))
    height, width = width, height
    points = tf.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
    points = tf.matmul(points, rotation_matrix) + [center_x, center_y]
    return points, angle

def intersection(gt_bboxes, default_bboxes):
    with tf.name_scope('bboxes_intersection', values=[gt_bboxes, default_bboxes]):
        # num_anchors x 1
        ymin, xmin, ymax, xmax = tf.split(gt_bboxes, 4, axis=1)
        # 1 x num_anchors
        gt_ymin, gt_xmin, gt_ymax, gt_xmax = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]
        # broadcast here to generate the full matrix
        int_ymin = tf.maximum(ymin, gt_ymin)
        int_xmin = tf.maximum(xmin, gt_xmin)
        int_ymax = tf.minimum(ymax, gt_ymax)
        int_xmax = tf.minimum(xmax, gt_xmax)
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        return h * w

def fill_poly(points, size):
    img = np.zeros((size,size))
    points = (points * size.astype(float)).astype(int)
    # points = np.stack([[int(pt[0] * size), int(pt[1] * size)] for pt in points], axis = 0)
    cv.fillPoly(img, [points], 1)
    return img

def get_box_mask( center_y, center_x, height, width):
    center_y, center_x, height, width = [tf.reshape(b, [-1]) for b in [center_y, center_x, height, width]]
    input_value = tf.stack([center_y, center_x, height, width], axis = 1)
    points, angle = tf.map_fn(center2point, input_value, (tf.float32, tf.float32))
    func = lambda points: tf.py_func(fill_poly, [points, 64], tf.float64)
    image = tf.cast(tf.map_fn(func, tf.cast(points, tf.float64)), tf.float32)
    return image


def iou_matrix2(gt_bboxes, default_bboxes, anchor_area):
    with tf.name_scope('iou_matrix', values=[gt_bboxes, default_bboxes]):
        # num_anchors x 1 
        gt_cy, gt_cx, gt_height, gt_width = tf.split(gt_bboxes, 4, axis=1)
        # 1 x num_anchors
        cy, cx, height, width = [tf.transpose(b, perm=[1, 0]) for b in tf.split(default_bboxes, 4, axis=1)]

        dist = tf.math.squared_difference(gt_cy, cy ) + tf.math.squared_difference(gt_cx,cx)
        size = tf.math.square(tf.maximum(tf.maximum(gt_height, gt_width), tf.maximum(height, width)))
        cond = tf.less(dist, size, name='cond_dist')
        mask1 = get_box_mask(gt_cy, gt_cx, gt_height, gt_width)
        # mask2 = get_box_mask(cy, cx, height, width)
        mask2 = anchor_area
        # return get_box_mask(cy, cx, height, width)[0]
        mask1 = tf.expand_dims(mask1, 1, name = 'mask1')
        mask2 = tf.expand_dims(mask2, 0, name = 'mask2')
        # inter_vol = tf.reduce_sum(tf.multiply(mask1, mask2, name = 'multiplyccc'), [2,3], name = 'reduce_sumfds')  
        # inter_vol = tf.cond(cond, lambda mask1, mask2: tf.ones_like(dist, type=tf.float32), lambda: tf.zeros_like(dist, type=tf.float32))
        # inter_vol = tf.where(cond, tf.ones_like(dist, dtype=tf.float32), tf.zeros_like(dist, dtype=tf.float32))
        inter_vol = tf.where(cond, tf.reduce_sum(tf.multiply(mask1, mask2), [2,3]), tf.zeros_like(dist, dtype=tf.float32))
        area1 = tf.reduce_sum(mask1, [2,3], name = 'reduce_sum1')
        area2 = tf.reduce_sum(mask2, [2,3], name = 'reduce_sum2')
        union_vol = area1 + area2 - inter_vol
        # image, points = get_box_mask(gt_cy, gt_cx, gt_height, gt_width)
        
        return tf.where(tf.equal(union_vol, 0.0),
                tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))


def iou_matrix(gt_bboxes, default_bboxes):
    with tf.name_scope('iou_matrix', values=[gt_bboxes, default_bboxes]):
        inter_vol = intersection(gt_bboxes, default_bboxes)
        # broadcast
        union_vol = areas(gt_bboxes) + tf.transpose(areas(default_bboxes), perm=[1, 0]) - inter_vol

        return tf.where(tf.equal(union_vol, 0.0),
                        tf.zeros_like(inter_vol), tf.truediv(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    with tf.name_scope('dual_max_match', values=[overlap_matrix]):
        # first match from anchors' side
        anchors_to_gt = tf.argmax(overlap_matrix, axis=0)
        # the matching degree
        match_values = tf.reduce_max(overlap_matrix, axis=0)

        #positive_mask = tf.greater(match_values, high_thres)
        less_mask = tf.less(match_values, low_thres)
        between_mask = tf.logical_and(tf.less(match_values, high_thres), tf.greater_equal(match_values, low_thres))
        negative_mask = less_mask if ignore_between else between_mask
        ignore_mask = between_mask if ignore_between else less_mask
        # fill all negative positions with -1, all ignore positions is -2
        match_indices = tf.where(negative_mask, -1 * tf.ones_like(anchors_to_gt), anchors_to_gt)
        match_indices = tf.where(ignore_mask, -2 * tf.ones_like(match_indices), match_indices)

        # negtive values has no effect in tf.one_hot, that means all zeros along that axis
        # so all positive match positions in anchors_to_gt_mask is 1, all others are 0
        anchors_to_gt_mask = tf.one_hot(tf.clip_by_value(match_indices, -1, tf.cast(tf.shape(overlap_matrix)[0], tf.int64)),
                                        tf.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=tf.int32)
        # match from ground truth's side
        gt_to_anchors = tf.argmax(overlap_matrix, axis=1)

        if gt_max_first:
            # the max match from ground truth's side has higher priority
            left_gt_to_anchors_mask = tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=tf.int32)
        else:
            # the max match from anchors' side has higher priority
            # use match result from ground truth's side only when the the matching degree from anchors' side is lower than position threshold
            left_gt_to_anchors_mask = tf.cast(tf.logical_and(tf.reduce_max(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                                                            tf.one_hot(gt_to_anchors, tf.shape(overlap_matrix)[1],
                                                                        on_value=True, off_value=False, axis=1, dtype=tf.bool)
                                                            ), tf.int64)
        # can not use left_gt_to_anchors_mask here, because there are many ground truthes match to one anchor, we should pick the highest one even when we are merging matching from ground truth side
        left_gt_to_anchors_scores = overlap_matrix * tf.to_float(left_gt_to_anchors_mask)
        # merge matching results from ground truth's side with the original matching results from anchors' side
        # then select all the overlap score of those matching pairs
        selected_scores = tf.gather_nd(overlap_matrix,  tf.stack([tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                                                                            tf.argmax(left_gt_to_anchors_scores, axis=0),
                                                                            anchors_to_gt),
                                                                    tf.range(tf.cast(tf.shape(overlap_matrix)[1], tf.int64))], axis=1))
        # return the matching results for both foreground anchors and background anchors, also with overlap scores
        return tf.where(tf.reduce_max(left_gt_to_anchors_mask, axis=0) > 0,
                        tf.argmax(left_gt_to_anchors_scores, axis=0),
                        match_indices), selected_scores

def save_anchor_area(area):
    if not hasattr(save_anchor_area, "counter"):
        save_anchor_area.counter = 0  # it doesn't exist yet, so initialize it
    save_anchor_area.counter += 1

    np.save('./debug/anchor_area_{}.npy'.format(save_anchor_area.counter), np.copy(area))
    return save_anchor_area.counter

def save_gt_value(gt_cy, gt_cx, gt_height, gt_width):
    if not hasattr(save_gt_value, "counter"):
        save_gt_value.counter = 0  # it doesn't exist yet, so initialize it
    save_gt_value.counter += 1
    np.savetxt('./debug/gt_cy_{}.txt'.format(save_gt_value.counter), gt_cy)
    np.savetxt('./debug/gt_cx_{}.txt'.format(save_gt_value.counter), gt_cx)
    np.savetxt('./debug/gt_height_{}.txt'.format(save_gt_value.counter), gt_height)
    np.savetxt('./debug/gt_width_{}.txt'.format(save_gt_value.counter), gt_width)
    return save_gt_value.counter


# def save_anchors(bboxes, labels, anchors_point):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     np.save('./debug/bboxes_{}.npy'.format(save_image_with_bbox.counter), np.copy(bboxes))
#     np.save('./debug/labels_{}.npy'.format(save_image_with_bbox.counter), np.copy(labels))
#     np.save('./debug/anchors_{}.npy'.format(save_image_with_bbox.counter), np.copy(anchors_point))
#     return save_image_with_bbox.counter

class AnchorEncoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling, anchor_area=None, clip=False):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling
        self._clip = clip
        self._anchor_area = anchor_area

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

    def center_duplicate(self, anchor):
        anchor_cy = tf.tile(anchor[0],[1,1,tf.size(anchor[2])])
        anchor_cx = tf.tile(anchor[1],[1,1,tf.size(anchor[3])])
        anchor_height = tf.tile(tf.expand_dims(tf.expand_dims(anchor[2],0),0),[tf.shape(anchor[0])[0], tf.shape(anchor[0])[1], 1])
        anchor_width =tf.tile(tf.expand_dims(tf.expand_dims(anchor[3],0),0),[tf.shape(anchor[1])[0], tf.shape(anchor[1])[1], 1])
        return anchor_cy, anchor_cx, anchor_height, anchor_width


    def _get_area(self, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('encode_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_cx = []
            list_anchors_cy = []
            list_anchors_w = []
            list_anchors_h = []
            tiled_allowed_borders = []
            with tf.name_scope('reshape_all_anchors'):
              for ind, anchor in enumerate(all_anchors):
                cy, cx, h, w = self.center_duplicate(anchor)
                list_anchors_cy.append(tf.reshape(cy, [-1]))
                list_anchors_cx.append(tf.reshape(cx, [-1]))
                list_anchors_h.append(tf.reshape(h, [-1]))
                list_anchors_w.append(tf.reshape(w, [-1]))
                tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

            anchors_cx = tf.concat(list_anchors_cx, 0, name='concat_cx') # Nx1
            anchors_cy = tf.concat(list_anchors_cy, 0, name='concat_cy')
            anchors_w = tf.concat(list_anchors_w, 0, name='concat_w')
            anchors_h = tf.concat(list_anchors_h, 0, name='concat_h')
 
            if self._clip:
                anchors_cx = tf.clip_by_value(anchors_cx, 0., 1.)
                anchors_cy = tf.clip_by_value(anchors_cy, 0., 1.)
                anchors_w = tf.clip_by_value(anchors_w, 0., 1.)
                anchors_h = tf.clip_by_value(anchors_h, 0., 1.)


            return get_box_mask(anchors_cy, anchors_cx, anchors_h, anchors_w)


    def encode_all_anchors(self, labels, bboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, debug=False):
        # y, x, h, w are all in range [0, 1] relative to the original image size
        # shape info:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors

        if self._anchor_area is None:
            self._anchor_area = self._get_area(all_anchors, all_num_anchors_depth, all_num_anchors_spatial)
            save_anchors_op = tf.py_func(save_anchor_area,
                            [self._anchor_area, ],
                            tf.int64, stateful=True)
            return save_anchors_op

        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('encode_all_anchors'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_cx = []
            list_anchors_cy = []
            list_anchors_w = []
            list_anchors_h = []
            tiled_allowed_borders = []
            with tf.name_scope('reshape_all_anchors'):
              for ind, anchor in enumerate(all_anchors):
                cy, cx, h, w = self.center_duplicate(anchor)
                list_anchors_cy.append(tf.reshape(cy, [-1]))
                list_anchors_cx.append(tf.reshape(cx, [-1]))
                list_anchors_h.append(tf.reshape(h, [-1]))
                list_anchors_w.append(tf.reshape(w, [-1]))
                tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

            anchors_cx = tf.concat(list_anchors_cx, 0, name='concat_cx') # Nx1
            anchors_cy = tf.concat(list_anchors_cy, 0, name='concat_cy')
            anchors_w = tf.concat(list_anchors_w, 0, name='concat_w')
            anchors_h = tf.concat(list_anchors_h, 0, name='concat_h')
 
            if self._clip:
                anchors_cx = tf.clip_by_value(anchors_cx, 0., 1.)
                anchors_cy = tf.clip_by_value(anchors_cy, 0., 1.)
                anchors_w = tf.clip_by_value(anchors_w, 0., 1.)
                anchors_h = tf.clip_by_value(anchors_h, 0., 1.)

            anchor_allowed_borders = tf.stack(tiled_allowed_borders, 0, name='concat_allowed_borders')

            inside_mask = tf.logical_and(tf.logical_and(anchors_cx > -anchor_allowed_borders * 1.,
                                                        anchors_cy > -anchor_allowed_borders * 1.),
                                        tf.logical_and(anchors_w < (1. + anchor_allowed_borders * 1.),
                                                        anchors_h < (1. + anchor_allowed_borders * 1.)))

            anchors_point = tf.stack([anchors_cy, anchors_cx, anchors_h, anchors_w], axis=-1) #Nx4
            # return iou_matrix2(bboxes, anchors_point) 

            
            # with tf.control_dependencies([save_anchors_op]):
            # save_anchors_op = tf.py_func(save_anchor_area,
            #                 [self._anchor_area, ],
            #                 tf.int64, stateful=True)
            # with tf.control_dependencies([save_anchors_op]):
            overlap_matrix = iou_matrix2(bboxes, anchors_point, self._anchor_area) * tf.cast(tf.expand_dims(inside_mask, 0), tf.float32)
            matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
            # get all positive matching positions
            matched_gt_mask = matched_gt > -1
            matched_indices = tf.clip_by_value(matched_gt, 0, tf.int64.max)
            # the labels here maybe chaos at those non-positive positions
            gt_labels = tf.gather(labels, matched_indices)
            # filter the invalid labels
            gt_labels = gt_labels * tf.cast(matched_gt_mask, tf.int64)
            # set those ignored positions to -1
            gt_labels = gt_labels + (-1 * tf.cast(matched_gt < -1, tf.int64))

            gt_cy, gt_cx, gt_height, gt_width = tf.unstack(tf.gather(bboxes, matched_indices), 4, axis=-1)

            # transform to center / size.
            # gt_cy, gt_cx, gt_h, gt_w = self.point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
            # anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            # encode features.
            # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
            gt_cy = (gt_cy - anchors_cy) / anchors_h / self._prior_scaling[0]
            gt_cx = (gt_cx - anchors_cx) / anchors_w / self._prior_scaling[1]
            gt_height = tf.log(gt_height / anchors_h) / self._prior_scaling[2]
            gt_width = tf.log(gt_width / anchors_w) / self._prior_scaling[3]

            # now gt_localizations is our regression object, but also maybe chaos at those non-positive positions
            if debug:
                gt_targets = tf.stack([anchors_cy, anchors_cx, anchors_h, anchors_w], axis=-1)
            else:
                gt_targets = tf.stack([gt_cy, gt_cx, gt_height, gt_width], axis=-1)
            # set all targets of non-positive positions to 0
            gt_targets_save = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets
            # save_gt_op = tf.py_func(save_gt_value,
            #                 [gt_cy, gt_targets_save, gt_height, gt_width ],
            #                 tf.int64, stateful=True) 
            # with tf.control_dependencies([save_gt_op]):
            gt_targets = tf.expand_dims(tf.cast(matched_gt_mask, tf.float32), -1) * gt_targets
            self._all_anchors = (anchors_cy, anchors_cx, gt_height, anchors_w)
            return gt_targets, gt_labels, gt_scores

    # return a list, of which each is:
    #   shape: [feature_h, feature_w, num_anchors, 4]
    #   order: ymin, xmin, ymax, xmax
    def decode_all_anchors(self, pred_location, num_anchors_per_layer):
        assert self._all_anchors is not None, 'no anchors to decode.'
        with tf.name_scope('decode_all_anchors', values=[pred_location]):
            anchor_cy, anchor_cx, anchor_h, anchor_w = self._all_anchors

            pred_h = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.split(tf.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

    def ext_decode_all_anchors(self, pred_location, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('ext_decode_all_anchors', values=[pred_location]):
            num_anchors_per_layer = []
            for ind in range(len(all_anchors)):
                num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])
            num_layers = len(all_num_anchors_depth)
            list_anchors_cx = []
            list_anchors_cy = []
            list_anchors_w = []
            list_anchors_h = []
            tiled_allowed_borders = []
            for ind, anchor in enumerate(all_anchors):
                cy, cx, h, w = self.center_duplicate(anchor)
                list_anchors_cy.append(tf.reshape(cy, [-1]))
                list_anchors_cx.append(tf.reshape(cx, [-1]))
                list_anchors_h.append(tf.reshape(h, [-1]))
                list_anchors_w.append(tf.reshape(w, [-1]))
            anchor_cx = tf.concat(list_anchors_cx, 0, name='concat_cx') # Nx1
            anchor_cy = tf.concat(list_anchors_cy, 0, name='concat_cy')
            anchor_w = tf.concat(list_anchors_w, 0, name='concat_w')
            anchor_h = tf.concat(list_anchors_h, 0, name='concat_h')


            pred_h = tf.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
            pred_w = tf.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
            pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
            pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

            return tf.split(tf.stack([pred_cy, pred_cx, pred_h, pred_w], axis=-1), num_anchors_per_layer, axis=0)



class AnchorCreator(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        super(AnchorCreator, self).__init__()
        # img_shape -> (height, width)
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)
        self._prior_scaling = prior_scaling
    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,
    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width
    def center_duplicate(self, anchor):
        anchor_cy = tf.tile(anchor[0],[1,1,tf.size(anchor[2])])
        anchor_cx = tf.tile(anchor[1],[1,1,tf.size(anchor[3])])
        anchor_height = tf.tile(tf.expand_dims(tf.expand_dims(anchor[2],0),0),[tf.shape(anchor[0])[0], tf.shape(anchor[0])[1], 1])
        anchor_width =tf.tile(tf.expand_dims(tf.expand_dims(anchor[3],0),0),[tf.shape(anchor[1])[0], tf.shape(anchor[1])[1], 1])
        return anchor_cy, anchor_cx, anchor_height, anchor_width
    def get_anchor_boxes(self, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, name='name'):

        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        with tf.name_scope('get_anchor_boxes'):
            num_layers = len(all_num_anchors_depth)
            list_anchors_cx = []
            list_anchors_cy = []
            list_anchors_w = []
            list_anchors_h = []
            tiled_allowed_borders = []
            with tf.name_scope('reshape_all_anchors'):
              for ind, anchor in enumerate(all_anchors):
                cy, cx, h, w = self.center_duplicate(anchor)
                list_anchors_cy.append(tf.reshape(cy, [-1]))
                list_anchors_cx.append(tf.reshape(cx, [-1]))
                list_anchors_h.append(tf.reshape(h, [-1]))
                list_anchors_w.append(tf.reshape(w, [-1]))
            anchors_cx = tf.concat(list_anchors_cx, 0, name='concat_cx') # Nx1
            anchors_cy = tf.concat(list_anchors_cy, 0, name='concat_cy')
            anchors_w = tf.concat(list_anchors_w, 0, name='concat_w')
            anchors_h = tf.concat(list_anchors_h, 0, name='concat_h')

            all_anchor_nums = tf.shape(anchors_cx)[0]
            # anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
            scale = tf.tile([self._prior_scaling], [all_anchor_nums ,1])
            box = tf.transpose(tf.reshape(tf.concat([anchors_cy, anchors_cx, anchors_h, anchors_w ], axis=-1), [4, -1]), [1, 0]) 
            prior_data = tf.concat([tf.reshape(box, [-1, 1, all_anchor_nums*4, 1]),
                                    tf.reshape(scale, [-1, 1, all_anchor_nums*4, 1])],
                                    axis = 1, name=name)
            return  prior_data[:,0,:,:]


    def get_layer_anchors(self, layer_shape, anchor_scale, extra_anchor_scale, anchor_ratio, layer_step, offset = 0.5):
        ''' assume layer_shape[0] = 6, layer_shape[1] = 5
        x_on_layer = [[0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4],
                       [0, 1, 2, 3, 4]]
        y_on_layer = [[0, 0, 0, 0, 0],
                       [1, 1, 1, 1, 1],
                       [2, 2, 2, 2, 2],
                       [3, 3, 3, 3, 3],
                       [4, 4, 4, 4, 4],
                       [5, 5, 5, 5, 5]]
        '''
        with tf.name_scope('get_layer_anchors'):
            x_on_layer, y_on_layer = tf.meshgrid(tf.range(layer_shape[1]), tf.range(layer_shape[0]))

            y_on_image = (tf.cast(y_on_layer, tf.float32) + offset) * layer_step / self._img_shape[0]
            x_on_image = (tf.cast(x_on_layer, tf.float32) + offset) * layer_step / self._img_shape[1]
            # print('anchor_scale = {}, {}'.format(len(anchor_scale), anchor_scale))
            # print('anchor_ratio = {}, {}'.format(len(anchor_ratio), anchor_ratio))
            num_anchors_along_depth = len(anchor_scale) * len(anchor_ratio) + len(extra_anchor_scale)
            num_anchors_along_spatial = layer_shape[1] * layer_shape[0]

            list_h_on_image = []
            list_w_on_image = []

            global_index = 0
            # for square anchors
            for _, scale in enumerate(extra_anchor_scale):
                list_h_on_image.append(scale)
                list_w_on_image.append(scale)
                global_index += 1
            # for other aspect ratio anchors
            for scale_index, scale in enumerate(anchor_scale):
                for ratio_index, ratio in enumerate(anchor_ratio):
                    list_h_on_image.append(scale / math.sqrt(ratio))
                    list_w_on_image.append(scale * math.sqrt(ratio))
                    global_index += 1
            # shape info:
            # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
            # h_on_image, w_on_image: num_anchors_along_depth
            return tf.expand_dims(y_on_image, axis=-1), tf.expand_dims(x_on_image, axis=-1), \
                    tf.constant(list_h_on_image, dtype=tf.float32), \
                    tf.constant(list_w_on_image, dtype=tf.float32), num_anchors_along_depth, num_anchors_along_spatial

    def get_all_anchors(self):
        all_anchors = []
        all_num_anchors_depth = []
        all_num_anchors_spatial = []
        for layer_index, layer_shape in enumerate(self._layers_shapes):
            anchors_this_layer = self.get_layer_anchors(layer_shape,
                                                        self._anchor_scales[layer_index],
                                                        self._extra_anchor_scales[layer_index],
                                                        self._anchor_ratios[layer_index],
                                                        self._layer_steps[layer_index],
                                                        self._anchor_offset[layer_index])
            all_anchors.append(anchors_this_layer[:-2])
            all_num_anchors_depth.append(anchors_this_layer[-2])
            all_num_anchors_spatial.append(anchors_this_layer[-1])
        return all_anchors, all_num_anchors_depth, all_num_anchors_spatial

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
        height, width = width, height
        points = tf.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
        points = tf.matmul(points, rotation_matrix) + [center_x, center_y]
        return points, angle

    def fill_poly(points, size):
        img = np.zeros((size,size))
        points = (points * size).astype(int)
        # points = np.stack([[int(pt[0] * size), int(pt[1] * size)] for pt in points], axis = 0)
        cv.fillPoly(img, [points], 128)
        return img 

    sess = tf.Session()
    init_op = tf.group([tf.local_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()])
    sess.run(init_op)
    
    def run_pt(y,x,h,w,name):
        center_x = tf.constant(x)
        center_y = tf.constant(y)
        width = tf.constant(w)
        height = tf.constant(h)
        center_y, center_x, height, width = [tf.reshape(b, [-1]) for b in [center_y, center_x, height, width]]
        
        # # # run map function: center size to 4 corner points
        input_value = tf.stack([center_y, center_x, height, width], axis = 1)
        points, angle = tf.map_fn(center2point, input_value, (tf.float32, tf.float32))

        func = lambda points: tf.py_func(fill_poly, [points, 128], tf.float64)
        # points = tf.constant([[[0.1,0.1],[0.6,0.3],[0.9,0.6],[0.2,0.7]],
        #                     [[0.1,0.8],[0.3,0.5],[0.7,0.5],[0.1,0.1]],
        #                     [[0.9,0.3],[0.1,0.6],[0.4,0.4],[0.5,0.2]]], tf.float64)
        image = tf.map_fn(func, tf.cast(points, tf.float64))
        image = sess.run(image)
        image.astype(np.uint8)
        for i in range(image.shape[0]):
            cv.imshow(name + str(i), image[i])

        size = 128
        points = tf.cast(tf.multiply(points, tf.constant(128.0)), tf.int32)
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
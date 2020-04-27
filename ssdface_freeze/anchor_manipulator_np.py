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
import numpy as np

from tensorflow.contrib.image.python.ops import image_ops

def one_hot(    indices,    depth,    on_value=1,    off_value=0,    axis=-1,    dtype=None,    name=None):
    # only valid for 1-D indices
    one_hot_matrix = np.eye(depth)[np.array(indices)]
    if(axis == 0):
        one_hot_matrix = np.transpose(one_hot_matrix)
    if not dtype == None:
        one_hot_matrix = one_hot_matrix.astype(dtype)

    return one_hot_matrix




def areas(gt_bboxes):
    ymin, xmin, ymax, xmax = np.split(gt_bboxes, 4, axis=1)
    return (xmax - xmin) * (ymax - ymin)

def intersection(gt_bboxes, default_bboxes):
    # num_anchors x 1
    ymin, xmin, ymax, xmax = np.split(gt_bboxes, 4, axis=1)
    # 1 x num_anchors
    gt_ymin, gt_xmin, gt_ymax, gt_xmax = [np.transpose(b, perm=[1, 0]) for b in np.split(default_bboxes, 4, axis=1)]
    # broadcast here to generate the full matrix
    int_ymin = max(ymin, gt_ymin)
    int_xmin = max(xmin, gt_xmin)
    int_ymax = min(ymax, gt_ymax)
    int_xmax = min(xmax, gt_xmax)
    h = max(int_ymax - int_ymin, 0.)
    w = max(int_xmax - int_xmin, 0.)
    return h * w

def iou_matrix(gt_bboxes, default_bboxes):
    inter_vol = intersection(gt_bboxes, default_bboxes)
    # broadcast
    union_vol = areas(gt_bboxes) + np.transpose(areas(default_bboxes)) - inter_vol
    return np.where(np.equal(union_vol, 0.0),
                    np.zeros_like(inter_vol), np.true_divide(inter_vol, union_vol))

def do_dual_max_match(overlap_matrix, low_thres, high_thres, ignore_between=True, gt_max_first=True):
    '''
    overlap_matrix: num_gt * num_anchors
    '''
    # first match from anchors' side
    anchors_to_gt = np.argmax(overlap_matrix, axis=0)
    # the matching degree
    match_values = np.amax(overlap_matrix, axis=0)

    #positive_mask = tf.greater(match_values, high_thres)
    less_mask = np.less(match_values, low_thres)
    between_mask = np.logical_and(np.less(match_values, high_thres), np.greater_equal(match_values, low_thres))
    negative_mask = less_mask if ignore_between else between_mask
    ignore_mask = between_mask if ignore_between else less_mask
    # fill all negative positions with -1, all ignore positions is -2
    match_indices = np.where(negative_mask, -1 * np.ones_like(anchors_to_gt), anchors_to_gt)
    match_indices = np.where(ignore_mask, -2 * np.ones_like(match_indices), match_indices)

    # negtive values has no effect in tf.one_hot, that means all zeros along that axis
    # so all positive match positions in anchors_to_gt_mask is 1, all others are 0
    anchors_to_gt_mask = one_hot(np.clip(match_indices, -1, np.shape(overlap_matrix)[0].astype(np.int64)),
                                    np.shape(overlap_matrix)[0], on_value=1, off_value=0, axis=0, dtype=np.int32)
    # match from ground truth's side
    gt_to_anchors = np.argmax(overlap_matrix, axis=1)

    if gt_max_first:
        # the max match from ground truth's side has higher priority
        left_gt_to_anchors_mask = one_hot(gt_to_anchors, np.shape(overlap_matrix)[1], on_value=1, off_value=0, axis=1, dtype=np.int32)
    else:
        # the max match from anchors' side has higher priority
        # use match result from ground truth's side only when the the matching degree from anchors' side is lower than position threshold
        left_gt_to_anchors_mask = np.logical_and(np.amax(anchors_to_gt_mask, axis=1, keep_dims=True) < 1,
                            one_hot(gt_to_anchors, np.shape(overlap_matrix)[1], on_value=True, off_value=False, axis=1, dtype=np.bool_)
                            ).astype(np.int64)
    # can not use left_gt_to_anchors_mask here, because there are many ground truthes match to one anchor, we should pick the highest one even when we are merging matching from ground truth side
    left_gt_to_anchors_scores = overlap_matrix * left_gt_to_anchors_mask.astype(np.float)
    # merge matching results from ground truth's side with the original matching results from anchors' side
    # then select all the overlap score of those matching pairs
    selected_scores = np.take(overlap_matrix,  np.stack([np.where(np.amax(left_gt_to_anchors_mask, axis=0) > 0,
                                                                        np.argmax(left_gt_to_anchors_scores, axis=0),
                                                                        anchors_to_gt),
                                                                range(np.shape(overlap_matrix)[1].astype(np.int64))], axis=1))
    # return the matching results for both foreground anchors and background anchors, also with overlap scores
    return np.where(np.amax(left_gt_to_anchors_mask, axis=0) > 0,
                    np.argmax(left_gt_to_anchors_scores, axis=0),
                    match_indices), selected_scores

# def save_anchors(bboxes, labels, anchors_point):
#     if not hasattr(save_image_with_bbox, "counter"):
#         save_image_with_bbox.counter = 0  # it doesn't exist yet, so initialize it
#     save_image_with_bbox.counter += 1

#     np.save('./debug/bboxes_{}.npy'.format(save_image_with_bbox.counter), np.copy(bboxes))
#     np.save('./debug/labels_{}.npy'.format(save_image_with_bbox.counter), np.copy(labels))
#     np.save('./debug/anchors_{}.npy'.format(save_image_with_bbox.counter), np.copy(anchors_point))
#     return save_image_with_bbox.counter

class AnchorEncoder(object):
    def __init__(self, allowed_borders, positive_threshold, ignore_threshold, prior_scaling, clip=False):
        super(AnchorEncoder, self).__init__()
        self._all_anchors = None
        self._allowed_borders = allowed_borders
        self._positive_threshold = positive_threshold
        self._ignore_threshold = ignore_threshold
        self._prior_scaling = prior_scaling
        self._clip = clip

    def center2point(self, center_y, center_x, height, width):
        return center_y - height / 2., center_x - width / 2., center_y + height / 2., center_x + width / 2.,

    def point2center(self, ymin, xmin, ymax, xmax):
        height, width = (ymax - ymin), (xmax - xmin)
        return ymin + height / 2., xmin + width / 2., height, width

    def encode_all_anchors(self, labels, bboxes, all_anchors, all_num_anchors_depth, all_num_anchors_spatial, debug=False):
        # y, x, h, w are all in range [0, 1] relative to the original image size
        # shape info:
        # y_on_image, x_on_image: layers_shapes[0] * layers_shapes[1]
        # h_on_image, w_on_image: num_anchors
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        
        num_layers = len(all_num_anchors_depth)
        list_anchors_ymin = []
        list_anchors_xmin = []
        list_anchors_ymax = []
        list_anchors_xmax = []
        tiled_allowed_borders = []
        for ind, anchor in enumerate(all_anchors):
            anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])

            list_anchors_ymin.append(np.reshape(anchors_ymin_, [-1]))
            list_anchors_xmin.append(np.reshape(anchors_xmin_, [-1]))
            list_anchors_ymax.append(np.reshape(anchors_ymax_, [-1]))
            list_anchors_xmax.append(np.reshape(anchors_xmax_, [-1]))

            tiled_allowed_borders.extend([self._allowed_borders[ind]] * all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

        anchors_ymin = np.concatenate(list_anchors_ymin, 0)
        anchors_xmin = np.concatenate(list_anchors_xmin, 0)
        anchors_ymax = np.concatenate(list_anchors_ymax, 0)
        anchors_xmax = np.concatenate(list_anchors_xmax, 0)

        if self._clip:
            anchors_ymin = np.clip(anchors_ymin, 0., 1.)
            anchors_xmin = np.clip(anchors_xmin, 0., 1.)
            anchors_ymax = np.clip(anchors_ymax, 0., 1.)
            anchors_xmax = np.clip(anchors_xmax, 0., 1.)

        anchor_allowed_borders = np.stack(tiled_allowed_borders, 0)

        inside_mask = np.logical_and(np.logical_and(anchors_ymin > -anchor_allowed_borders * 1.,
                                                    anchors_xmin > -anchor_allowed_borders * 1.),
                                    np.logical_and(anchors_ymax < (1. + anchor_allowed_borders * 1.),
                                                    anchors_xmax < (1. + anchor_allowed_borders * 1.)))

        anchors_point = np.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)

        # save_anchors_op = tf.py_func(save_anchors,
        #                 [bboxes,
        #                 labels,
        #                 anchors_point],
        #                 tf.int64, stateful=True)

        # with tf.control_dependencies([save_anchors_op]):
        overlap_matrix = iou_matrix(bboxes, anchors_point) * np.expand_dims(inside_mask, 0).astype(np.float32)
        matched_gt, gt_scores = do_dual_max_match(overlap_matrix, self._ignore_threshold, self._positive_threshold)
        # get all positive matching positions
        matched_gt_mask = matched_gt > -1
        matched_indices = np.clip(matched_gt, 0, np.iinfo(np.int64).max)
        # the labels here maybe chaos at those non-positive positions
        gt_labels = np.take(labels, matched_indices)
        # filter the invalid labels
        gt_labels = gt_labels * matched_gt_mask.astype(np.int64)
        # set those ignored positions to -1
        gt_labels = gt_labels + (-1 * (matched_gt < -1).astype(np.int64))

        gt_ymin, gt_xmin, gt_ymax, gt_xmax = np.unstack(np.take(bboxes, matched_indices), 4, axis=-1)

        # transform to center / size.
        gt_cy, gt_cx, gt_h, gt_w = self.point2center(gt_ymin, gt_xmin, gt_ymax, gt_xmax)
        anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
        # encode features.
        # the prior_scaling (in fact is 5 and 10) is use for balance the regression loss of center and with(or height)
        gt_cy = (gt_cy - anchor_cy) / anchor_h / self._prior_scaling[0]
        gt_cx = (gt_cx - anchor_cx) / anchor_w / self._prior_scaling[1]
        gt_h = math.log(gt_h / anchor_h) / self._prior_scaling[2]
        gt_w = math.log(gt_w / anchor_w) / self._prior_scaling[3]
        # now gt_localizations is our regression object, but also maybe chaos at those non-positive positions
        if debug:
            gt_targets = np.stack([anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax], axis=-1)
        else:
            gt_targets = np.stack([gt_cy, gt_cx, gt_h, gt_w], axis=-1)
        # set all targets of non-positive positions to 0
        gt_targets = np.expand_dims(matched_gt_mask.astype(np.float32), -1) * gt_targets
        self._all_anchors = (anchor_cy, anchor_cx, anchor_h, anchor_w)
        return gt_targets, gt_labels, gt_scores

    # return a list, of which each is:
    #   shape: [feature_h, feature_w, num_anchors, 4]
    #   order: ymin, xmin, ymax, xmax
    def decode_all_anchors(self, pred_location, num_anchors_per_layer):
        assert self._all_anchors is not None, 'no anchors to decode.'
        anchor_cy, anchor_cx, anchor_h, anchor_w = self._all_anchors

        pred_h = math.exp(pred_location[:, -2] * self._prior_scaling[2]) * anchor_h
        pred_w = math.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
        pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
        pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx

        return np.split(np.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), num_anchors_per_layer, axis=0)

    def ext_decode_all_anchors(self, pred_location, all_anchors, all_num_anchors_depth, all_num_anchors_spatial):
        assert (len(all_num_anchors_depth)==len(all_num_anchors_spatial)) and (len(all_num_anchors_depth)==len(all_anchors)), 'inconsist num layers for anchors.'
        num_anchors_per_layer = []
        for ind in range(len(all_anchors)):
            num_anchors_per_layer.append(all_num_anchors_depth[ind] * all_num_anchors_spatial[ind])

        num_layers = len(all_num_anchors_depth)
        list_anchors_ymin = []
        list_anchors_xmin = []
        list_anchors_ymax = []
        list_anchors_xmax = []
        tiled_allowed_borders = []
        for ind, anchor in enumerate(all_anchors):

            anchors_ymin_, anchors_xmin_, anchors_ymax_, anchors_xmax_ = self.center2point(anchor[0], anchor[1], anchor[2], anchor[3])

            list_anchors_ymin.append(np.reshape(anchors_ymin_, [-1]))
            list_anchors_xmin.append(np.reshape(anchors_xmin_, [-1]))
            list_anchors_ymax.append(np.reshape(anchors_ymax_, [-1]))
            list_anchors_xmax.append(np.reshape(anchors_xmax_, [-1]))
            # if(ind > 1):
            #     print('\n==\n anchors_ymin_ = {}, anchors_ymax_ = {}'.format(np.reshape(anchors_ymin_, [-1]),np.reshape(anchors_ymax_, [-1])))
        anchors_ymin = np.concatenate(list_anchors_ymin, 0)
        anchors_xmin = np.concatenate(list_anchors_xmin, 0)
        anchors_ymax = np.concatenate(list_anchors_ymax, 0)
        anchors_xmax = np.concatenate(list_anchors_xmax, 0)

        anchor_cy, anchor_cx, anchor_h, anchor_w = self.point2center(anchors_ymin, anchors_xmin, anchors_ymax, anchors_xmax)
        selected = [318, 330, 368, 369, 379]
        # print("\n==\n selected anchor = \n{}\n{}\n{}\n{}".format(np.take(anchor_cy, selected), np.take(anchor_cx, selected), 
        #         np.take(anchor_h, selected), np.take(anchor_w, selected)))
        # print("\n==\n pred_offset = \n{}".format(np.take(pred_location, selected, axis=0)))

        pred_h = np.exp(pred_location[:,-2] * self._prior_scaling[2]) * anchor_h
        pred_w = np.exp(pred_location[:, -1] * self._prior_scaling[3]) * anchor_w
        pred_cy = pred_location[:, 0] * self._prior_scaling[0] * anchor_h + anchor_cy
        pred_cx = pred_location[:, 1] * self._prior_scaling[1] * anchor_w + anchor_cx
        # # print("\n==\n pred cy cx h w = \n{}\n{}\n{}\n{}".format(np.take(pred_cy, selected), np.take(pred_cx, selected), 
        #         np.take(pred_h, selected), np.take(pred_w, selected)))        
        split_param = []
        split_param.append(num_anchors_per_layer[0])
        for layer_num in num_anchors_per_layer[1:-1]:
            split_param.append(layer_num + split_param[-1])

        return np.split(np.stack(self.center2point(pred_cy, pred_cx, pred_h, pred_w), axis=-1), split_param, axis=0)

class AnchorCreator(object):
    def __init__(self, img_shape, layers_shapes, anchor_scales, extra_anchor_scales, anchor_ratios, layer_steps):
        super(AnchorCreator, self).__init__()
        # img_shape -> (height, width)
        self._img_shape = img_shape
        self._layers_shapes = layers_shapes
        self._anchor_scales = anchor_scales
        self._extra_anchor_scales = extra_anchor_scales
        self._anchor_ratios = anchor_ratios
        self._layer_steps = layer_steps
        self._anchor_offset = [0.5] * len(self._layers_shapes)

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
        x_on_layer, y_on_layer = np.meshgrid(range(layer_shape[1]), range(layer_shape[0]))

        y_on_image = (y_on_layer.astype(np.float32) + offset) * layer_step / self._img_shape[0]
        x_on_image = (x_on_layer.astype(np.float32) + offset) * layer_step / self._img_shape[1]
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

        print('list_h_on_image = {}, list_w_on_image = {}'.format(list_h_on_image, list_w_on_image))
        return np.expand_dims(y_on_image, axis=-1), np.expand_dims(x_on_image, axis=-1), \
                np.array(list_h_on_image), \
                np.array(list_w_on_image), num_anchors_along_depth, num_anchors_along_spatial

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


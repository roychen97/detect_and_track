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

import tensorflow as tf
import numpy as np

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
_USE_FUSED_BN = True



def __variable_with_weight_decay(kernel_shape, initializer, wd):
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)
    return w

class MobileNetV2(object):
    def __init__(self, size, data_format='channels_first'):
        super(MobileNetV2, self).__init__()
        self._data_format = data_format
        self._bn_axis = -1 if data_format == 'channels_last' else 1
        #initializer = tf.glorot_uniform_initializer  glorot_normal_initializer
        self._conv_initializer = tf.glorot_uniform_initializer
        self._conv_bn_initializer = tf.glorot_uniform_initializer
        self.l2_strength = 5e-6
        self.training = False
        self.input_size = size
    #     self.img_mean = self.__init_mean()

    # def __init_mean(self):
    #     # Preparing the mean image.
    #     img_mean = np.ones((1, self.input_size, self.input_size, 3))
    #     img_mean[:, :, :, 0] *= 128.0
    #     img_mean[:, :, :, 1] *= 128.0
    #     img_mean[:, :, :, 2] *= 128.0
    #     if  self._data_format == 'channels_first':
    #         img_mean = np.transpose(img_mean, [0,3,1,2])
    #     return tf.constant(img_mean, dtype=tf.float32)

    def forward(self, inputs, training=False):
        # inputs = (inputs - self.img_mean) * 2.0 / 255.0
        self.training = training
        # inputs should in BGR
        feature_layers = []
        # 300
        inputs = self._conv2d('Conv', inputs, filters=32, kernel=3, stride=2, activation=tf.nn.relu6)
        # 150
        # feature_layers.append(inputs)
        inputs = self.expand_blocks('expanded_conv', inputs, kernel=3, filters=16, stride=1, k=None, 
            residual = False)
        inputs = self.expand_blocks('expanded_conv_1', inputs, filters=24, stride=2, residual=False)
        inputs = self.expand_blocks('expanded_conv_2', inputs, filters=24, stride=1, residual=True)
        # feature_layers.append(inputs) 2304
        # 75
        inputs = self.expand_blocks('expanded_conv_3', inputs, filters=32, stride=2, residual=False)
        inputs = self.expand_blocks('expanded_conv_4', inputs, filters=32, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_5', inputs, filters=32, stride=1, residual=True)
        # 38
        inputs = self.expand_blocks('expanded_conv_6', inputs, filters=64, stride=2, residual=False)
        inputs = self.expand_blocks('expanded_conv_7', inputs, filters=64, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_8', inputs, filters=64, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_9', inputs, filters=64, stride=1, residual=True)
        # 19
        inputs = self.expand_blocks('expanded_conv_10', inputs, filters=96, stride=1, residual=False)
        inputs = self.expand_blocks('expanded_conv_11', inputs, filters=96, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_12', inputs, filters=96, stride=1, residual=True)
        feature_layers.append(inputs)
        # 19
        inputs = self.expand_blocks('expanded_conv_13', inputs, filters=160, stride=2, residual=False)
        inputs = self.expand_blocks('expanded_conv_14', inputs, filters=160, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_15', inputs, filters=160, stride=1, residual=True)
        inputs = self.expand_blocks('expanded_conv_16', inputs, filters=320, stride=1, residual=False)
        # 10
        inputs = self._conv2d('Conv_1', inputs, filters=1280, kernel=1, stride=1, activation=tf.nn.relu6)
        feature_layers.append(inputs)

        inputs = self._conv2d('layer_19_1_Conv2d_2_1x1_256', inputs, filters=256, kernel=1, stride=1, activation=tf.nn.relu6)
        inputs = self._dw_conv2('layer_19_2_Conv2d_2_3x3_s2_512_depthwise', inputs, kernel=3, stride=2,activation=tf.nn.relu6)
        inputs = self._conv2d('layer_19_2_Conv2d_2_3x3_s2_512', inputs, filters=512, kernel=1, stride=1, activation=tf.nn.relu6)
        feature_layers.append(inputs)
        #5

        inputs = self._conv2d('layer_19_1_Conv2d_3_1x1_128', inputs, filters=128, kernel=1, stride=1, activation=tf.nn.relu6)
        inputs = self._dw_conv2('layer_19_2_Conv2d_3_3x3_s2_256_depthwise', inputs, kernel=3, stride=2, activation=tf.nn.relu6)
        inputs = self._conv2d('layer_19_2_Conv2d_3_3x3_s2_256', inputs, filters=256, kernel=1, stride=1, activation=tf.nn.relu6)
        feature_layers.append(inputs)
        #3

        inputs = self._conv2d('layer_19_1_Conv2d_4_1x1_128', inputs, filters=128, kernel=1, stride=1, activation=tf.nn.relu6)
        inputs = self._dw_conv2('layer_19_2_Conv2d_4_3x3_s2_256_depthwise', inputs, kernel=3, stride=2, activation=tf.nn.relu6)
        inputs = self._conv2d('layer_19_2_Conv2d_4_3x3_s2_256', inputs, filters=256, kernel=1, stride=1, activation=tf.nn.relu6)
        feature_layers.append(inputs)
        #2

        inputs = self._conv2d('layer_19_1_Conv2d_5_1x1_64', inputs, filters=64, kernel=1, stride=1, activation=tf.nn.relu6)
        inputs = self._dw_conv2('layer_19_2_Conv2d_5_3x3_s2_128_depthwise', inputs, kernel=3, stride=2,activation=tf.nn.relu6)
        inputs = self._conv2d('layer_19_2_Conv2d_5_3x3_s2_128', inputs, filters=128, kernel=1, stride=1, activation=tf.nn.relu6)
        feature_layers.append(inputs)
        #1      
  

        return feature_layers

    def l2_normalize(self, x, name):
        with tf.name_scope(name, "l2_normalize", [x]) as name:
            axis = -1 if self._data_format == 'channels_last' else 1
            square_sum = tf.reduce_sum(tf.square(x), axis, keep_dims=True)
            square_sum = tf.clip_by_value(square_sum, 1e-10, 1e10)
            x_inv_norm = tf.rsqrt(square_sum)
            return tf.multiply(x, x_inv_norm, name=name)

    def expand_blocks(self, name, x, kernel=3, filters=96, stride=1, k=6, residual=False ):
        with tf.variable_scope(name):
          if k == None:
            conv_1 = x
          else:
            ch_expand = x.get_shape()[self._bn_axis].value * 6
            conv_1 = self._conv2d('expand', x=x, filters=ch_expand, kernel=1, activation=tf.nn.relu6)
          # depthwise convolution
          conv_2 = self._dw_conv2('depthwise', conv_1, kernel=kernel, stride=stride, activation=tf.nn.relu6)
          # project
          conv_3 = self._conv2d('project', x=conv_2, filters=filters, kernel=1)
          if residual:
            conv_o = tf.math.add(conv_3, x)
          else:
            conv_o = conv_3
          return conv_o

    def _conv2d(self, name, x, w=None, filters=96, kernel=3, stride=1, bias=None,
        padding='SAME', activation=None, batchnorm_enabled=True):
        with tf.variable_scope(name) as scope:
            kernel_shape = [kernel, kernel, x.shape[self._bn_axis], filters]
            # prepare weight
            with tf.name_scope('layer_weights'):
                if w == None:
                    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                    #w = __variable_with_weight_decay(kernel_shape, tf.contrib.layers.xavier_initializer(), l2_strength)
            with tf.name_scope('layer_biases'):
              if not bias == None:
                if isinstance(bias, float):
                    bias = tf.get_variable('biases', [filters], initializer=tf.constant_initializer(bias))
            # conv2d
            with tf.name_scope('layer_conv2d'):
                if self._data_format=='channels_first':
                    #print('conv 2d stride={}'.format(stride))
                    conv = tf.nn.conv2d(x, w, stride, padding, data_format='NCHW')
                else:
                    conv = tf.nn.conv2d(x, w, stride, padding)
                if not bias == None:
                  out = tf.nn.bias_add(conv, bias)
                else:
                  out = conv
            # batchnorm & activation
            if batchnorm_enabled:
                #conv_o_bn = tf.layers.batch_normalization( conv_o_b, training=is_training)
                
                if self._data_format=='channels_first':
                    conv_o_bn = tf.contrib.layers.batch_norm( out, fused=False, is_training=self.training, data_format='NCHW')
                else:
                    conv_o_bn = tf.layers.batch_normalization( out, name="BatchNorm", fused=False, training=self.training)

                if not activation:
                    conv_a = conv_o_bn
                else:
                    conv_a = activation(conv_o_bn)
            else:
                if not activation:
                    conv_a = out
                else:
                    conv_a = activation(out)
            return conv_a

    def _dw_conv2(self,name,  x, w=None, kernel=3, stride=1, bias=None,
        padding='SAME', activation=None, batchnorm_enabled=True):
        with tf.variable_scope(name):

            kernel_shape = [kernel, kernel, x.shape[self._bn_axis], 1]
            with tf.name_scope('layer_weights'):
                if w is None:
                    w = tf.get_variable('depthwise_weights', kernel_shape, tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            with tf.name_scope('layer_biases'):
              if not bias == None:
                if isinstance(bias, float):
                    bias = tf.get_variable('biases', [x.shape[self._bn_axis]], initializer=tf.constant_initializer(bias))

            with tf.name_scope('layer_conv2d'):
                if self._data_format=='channels_first':
                    stride = [1, 1, stride, stride]
                    #print('dw convsd stride={}'.format(stride))
                    conv = tf.nn.depthwise_conv2d(x, w, stride, padding, data_format='NCHW')
                else:
                    stride = [1, stride, stride, 1]
                    conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
                if not bias == None:
                  out = tf.nn.bias_add(conv, bias)
                else:
                  out = conv
            # batchnorm & activation
            if batchnorm_enabled:
                #conv_o_bn = tf.layers.batch_normalization( conv_o_b, training=is_training)
                # conv_o_bn = tf.layers.batch_normalization( out, name="BatchNorm", fused=False, training=self.training, data_format='NCHW')
                if self._data_format=='channels_first':
                    conv_o_bn = tf.contrib.layers.batch_norm( out, fused=False, is_training=self.training, data_format='NCHW')
                else:
                    conv_o_bn = tf.layers.batch_normalization( out, name="BatchNorm", fused=False, training=self.training)

                if not activation:
                    conv_a = conv_o_bn
                else:
                    conv_a = activation(conv_o_bn)
            else:
                if not activation:
                    conv_a = out
                else:
                    conv_a = activation(out)
            return conv_a


def multibox_head(feature_layers, num_classes, num_anchors_depth_per_layer, data_format='channels_first'):
    with tf.variable_scope('multibox_head'):
        cls_preds = []
        loc_preds = []
        for ind, feat in enumerate(feature_layers):
            loc_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * 4, (3, 3), use_bias=True,
                        name='loc_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))
            cls_preds.append(tf.layers.conv2d(feat, num_anchors_depth_per_layer[ind] * num_classes, (3, 3), use_bias=True,
                        name='cls_{}'.format(ind), strides=(1, 1),
                        padding='same', data_format=data_format, activation=None,
                        kernel_initializer=tf.glorot_uniform_initializer(),
                        bias_initializer=tf.zeros_initializer()))

        return loc_preds, cls_preds

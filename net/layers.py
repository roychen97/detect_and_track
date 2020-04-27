import tensorflow as tf
import numpy as np

############################################################################################################
# Convolution layer Methods (MobileNetV2)
def expanded_conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', 
                    stride=(1, 1), k=None, depth_multiplier=1.0, residual=False,
                    initializer=tf.contrib.layers.xavier_initializer(), 
                    l2_strength=0.0, bias=(0.0, 0.0, 0.0), activation=None,
                    batchnorm_enabled=False, dropout_keep_prob=-1, is_training=True):
    with tf.variable_scope(name):
      # expand (rate = k)
      if k == None:
        conv_1 = x
      else:
        ch_expand = x.get_shape()[-1].value * 6
        conv_1 = conv2d('expand', x=x, w=None, num_filters=ch_expand, kernel_size=(1, 1),
                      initializer=initializer, l2_strength=l2_strength, bias=None, activation=activation,
                      batchnorm_enabled=batchnorm_enabled, dropout_keep_prob=dropout_keep_prob, is_training=is_training)      
      # depthwise convolution
      conv_2 = depthwise_conv2d('depthwise', x=conv_1, w=None, kernel_size=kernel_size, padding=padding,stride=stride,
                    initializer=initializer, l2_strength=l2_strength, bias=None, activation=activation,
                    batchnorm_enabled=batchnorm_enabled, is_training=is_training)
      # project
      conv_3 = conv2d('project', x=conv_2, w=None, num_filters=num_filters, kernel_size=(1, 1),
                    initializer=initializer, l2_strength=l2_strength, bias=None, activation=None,
                    batchnorm_enabled=batchnorm_enabled,  dropout_keep_prob=dropout_keep_prob, is_training=is_training)      
      if residual:
        conv_o = conv_3 + x
      else:
        conv_o = conv_3
      return conv_o
      
############################################################################################################
# Convolution layer Methods
def __conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    """
    Convolution 2D Wrapper
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param w: (tf.tensor) pretrained weights (if None, it means no pretrained weights)
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H', W', num_filters)
    """
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]
        with tf.name_scope('layer_weights'):
            if w == None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
          if not bias == None:
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            if not bias == None:
              out = tf.nn.bias_add(conv, bias)
            else:
              out = conv
    return out


def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    """
    This block is responsible for a convolution 2D layer followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, H, W, C).
    :param num_filters: (integer) No. of filters (This is the output depth)
    :param kernel_size: (integer tuple) The size of the convolving kernel.
    :param padding: (string) The amount of padding required.
    :param stride: (integer tuple) The stride required.
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param max_pool_enabled:  (boolean) for enabling max-pooling 2x2 to decrease width and height by a factor of 2.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return: The output tensor of the layer (N, H', W', C').
    """
    with tf.variable_scope(name) as scope:
        conv_o_b = __conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            #conv_o_bn = tf.layers.batch_normalization( conv_o_b, training=is_training)
            conv_o_bn = tf.layers.batch_normalization( conv_o_b, name="BatchNorm", training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)/dropout_keep_prob

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o


def __depthwise_conv2d_p(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                         initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], 1]

        with tf.name_scope('layer_weights'):
            if w is None:
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
          if not bias == None:
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [x.shape[-1]], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.depthwise_conv2d(x, w, stride, padding)
            if not bias == None:
              out = tf.nn.bias_add(conv, bias)
            else:
              out = conv

    return out


def depthwise_conv2d(name, x, w=None, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
                     initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0, activation=None,
                     batchnorm_enabled=False, is_training=True):
    """Implementation of depthwise 2D convolution wrapper"""
    with tf.variable_scope(name) as scope:
        conv_o_b = __depthwise_conv2d_p(name=scope, x=x, w=w, kernel_size=kernel_size, padding=padding,
                                        stride=stride, initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            #conv_o_bn = tf.layers.batch_normalization( conv_o_b, training=is_training)
            conv_o_bn = tf.layers.batch_normalization( conv_o_b, name="BatchNorm", training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)
    return conv_a


def depthwise_separable_conv2d(name, x, w_depthwise=None, w_pointwise=None, width_multiplier=1.0, num_filters=16,
                               kernel_size=(3, 3),
                               padding='SAME', stride=(1, 1),
                               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                               activation=None, batchnorm_enabled=True,
                               is_training=True):
    """Implementation of depthwise separable 2D convolution operator as in MobileNet paper"""
    total_num_filters = int(round(num_filters * width_multiplier))
    with tf.variable_scope(name) as scope:
        conv_a = depthwise_conv2d('depthwise', x=x, w=w_depthwise, kernel_size=kernel_size, padding=padding,
                                  stride=stride,
                                  initializer=initializer, l2_strength=l2_strength, bias=biases[0],
                                  activation=activation,
                                  batchnorm_enabled=batchnorm_enabled, is_training=is_training)

        conv_o = conv2d('pointwise', x=conv_a, w=w_pointwise, num_filters=total_num_filters, kernel_size=(1, 1),
                        initializer=initializer, l2_strength=l2_strength, bias=biases[1], activation=activation,
                        batchnorm_enabled=batchnorm_enabled, is_training=is_training)

    return conv_a, conv_o


############################################################################################################
# Fully Connected layer Methods

def __dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
              bias=0.0):
    """
    Fully connected layer
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias. (if not float, it means pretrained bias)
    :return out: The output of the layer. (N, H)
    """
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = __variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        __variable_summaries(w)
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        __variable_summaries(bias)
        if not bias == None:
            output = tf.nn.bias_add(tf.matmul(x, w), bias)
        else:
            output = tf.matmul(x, w)
        return output


def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    """
    This block is responsible for a fully connected followed by optional (non-linearity, dropout, max-pooling).
    Note that: "is_training" should be passed by a correct value based on being in either training or testing.
    :param name: (string) The name scope provided by the upper tf.name_scope('name') as scope.
    :param x: (tf.tensor) The input to the layer (N, D).
    :param output_dim: (integer) It specifies H, the output second dimension of the fully connected layer [ie:(N, H)]
    :param initializer: (tf.contrib initializer) The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param l2_strength:(weight decay) (float) L2 regularization parameter.
    :param bias: (float) Amount of bias.
    :param activation: (tf.graph operator) The activation function applied after the convolution operation. If None, linear is applied.
    :param batchnorm_enabled: (boolean) for enabling batch normalization.
    :param dropout_keep_prob: (float) for the probability of keeping neurons. If equals -1, it means no dropout
    :param is_training: (boolean) to diff. between training and testing (important for batch normalization and dropout)
    :return out: The output of the layer. (N, H)
    """
    with tf.variable_scope(name) as scope:
        dense_o_b = __dense_p(name=scope, x=x, w=w, output_dim=output_dim, initializer=initializer,
                              l2_strength=l2_strength,
                              bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(dense_a, dropout_keep_prob)/dropout_keep_prob

        def dropout_no_keep():
            return tf.nn.dropout(dense_a, 1.0)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def dropout(x, dropout_keep_prob, is_training):
    """Dropout special layer"""

    def dropout_with_keep():
        return tf.nn.dropout(x, dropout_keep_prob)/dropout_keep_prob

    def dropout_no_keep():
        return tf.nn.dropout(x, 1.0)

    if dropout_keep_prob != -1:
        output = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
    else:
        output = x
    return output


def flatten(x):
    """
    Flatten a (N,H,W,C) input into (N,D) output. Used for fully connected layers after conolution layers
    :param x: (tf.tensor) representing input
    :return: flattened output
    """
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o


############################################################################################################
# Pooling Methods

def max_pool_2d(x, size=(2, 2), stride=(2, 2), name='pooling'):
    """
    Max pooling 2D Wrapper
    :param x: (tf.tensor) The input to the layer (N,H,W,C).
    :param size: (tuple) This specifies the size of the filter as well as the stride.
    :param stride: (tuple) specifies the stride of pooling.
    :param name: (string) Scope name.
    :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.max_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


def avg_pool_2d(x, size=(2, 2), stride=(2, 2), name='avg_pooling'):
    """
        Average pooling 2D Wrapper
        :param x: (tf.tensor) The input to the layer (N,H,W,C).
        :param size: (tuple) This specifies the size of the filter as well as the stride.
        :param name: (string) Scope name.
        :return: The output is the same input but halfed in both width and height (N,H/2,W/2,C).
    """
    size_x, size_y = size
    stride_x, stride_y = stride
    return tf.nn.avg_pool(x, ksize=[1, size_x, size_y, 1], strides=[1, stride_x, stride_y, 1], padding='VALID',
                          name=name)


############################################################################################################
# Utilities for layers

def __variable_with_weight_decay(kernel_shape, initializer, wd):
    """
    Create a variable with L2 Regularization (Weight Decay)
    :param kernel_shape: the size of the convolving weight kernel.
    :param initializer: The initialization scheme, He et al. normal or Xavier normal are recommended.
    :param wd:(weight decay) L2 regularization parameter.
    :return: The weights of the kernel initialized. The L2 loss is added to the loss collection.
    """
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    return w


# Summaries for variables
def __variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: variable to be summarized
    :return: None
    """
    return
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



###########################################################################################
def Hour_Glass(name, x, w=None, n = 1, output_ch=256, is_training=True):
    with tf.variable_scope(name):

        x_mp = max_pool_2d(x)
        low1 = conv_block('conv1', x_mp, output_ch=output_ch, is_training=is_training)
        if n > 1:
            low2 = Hour_Glass(name + '_' + str(n), low1, n = n-1, output_ch=output_ch, is_training=is_training)

        else:
            low2 = conv_block('conv2', low1, output_ch=output_ch, is_training=is_training)
        low3 = conv_block('conv3', low2, output_ch=output_ch, is_training=is_training)

        up2 = tf.image.resize(low3, [tf.cast(tf.shape(x)[1], tf.int32), tf.cast(tf.shape(x)[2], tf.int32)])
        up1 = conv_block('conv_res', x, output_ch=output_ch, is_training=is_training)
        return up1 + up2


def conv_block(name, x, w=None, output_ch=128, is_training=True):
    with tf.variable_scope(name):
        x_0 = tf.nn.relu6(tf.layers.batch_normalization(x, training=is_training))        
        conv1_dw, conv1_pw = depthwise_separable_conv2d('conv_ds_1', x_0, w_depthwise=None, w_pointwise=None, 
                           width_multiplier=1.0, 
                           num_filters=output_ch/2,
                           kernel_size=(3, 3),
                           padding='SAME', stride=(1, 1),
                           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                           activation=None, batchnorm_enabled=False,
                           is_training=is_training)
            
        x_1 = tf.nn.relu6(tf.layers.batch_normalization(conv1_pw, training=is_training))
        conv2_dw, conv2_pw = depthwise_separable_conv2d('conv_ds_2', x_1, w_depthwise=None, w_pointwise=None, 
                           width_multiplier=1.0, 
                           num_filters=output_ch/4,
                           kernel_size=(3, 3),
                           padding='SAME', stride=(1, 1),
                           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                           activation=None, batchnorm_enabled=False,
                           is_training=is_training)      
      
        x_2 = tf.nn.relu6(tf.layers.batch_normalization(conv2_pw, training=is_training))
        conv3_dw, conv3_pw = depthwise_separable_conv2d('conv_ds_3', x_2, w_depthwise=None, w_pointwise=None, 
                           width_multiplier=1.0, 
                           num_filters=output_ch/4,
                           kernel_size=(3, 3),
                           padding='SAME', stride=(1, 1),
                           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                           activation=None, batchnorm_enabled=False,
                           is_training=is_training) 
        conv_cat = tf.concat([conv1_pw, conv2_pw, conv3_pw], 3)
        
        if tf.shape(x)[3] != output_ch:
            x_res_0 = tf.nn.relu6(tf.layers.batch_normalization(x, training=is_training))        
            x_res_1_dw, x_res_1_pw = depthwise_separable_conv2d('conv_ds_res1', x_res_0, w_depthwise=None, w_pointwise=None, 
                           width_multiplier=1.0, 
                           num_filters=output_ch,
                           kernel_size=(3, 3),
                           padding='SAME', stride=(1, 1),
                           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, biases=(0.0, 0.0),
                           activation=None, batchnorm_enabled=False,
                           is_training=is_training)
        else:
            x_res_1_pw = x
        return conv_cat + x_res_1_pw


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
           
def bilinear_upsample_weights(factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with bilinear filter
    initialization.
    """
    
    filter_size = get_kernel_size(factor)
    
    weights = np.zeros((filter_size,
                        filter_size,
                        number_of_classes,
                        number_of_classes), dtype=np.float32)
    
    upsample_kernel = upsample_filt(filter_size)
    
    for i in xrange(number_of_classes):
        
        weights[:, :, i, i] = upsample_kernel
    
    return tf.convert_to_tensor(weights)


def __deconv2d_p(name, x, output_shape, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
               initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], num_filters, x.shape[-1]]

        with tf.name_scope('layer_weights'):
            if w == None:
                #w = bilinear_upsample_weights(stride[0], num_filters)
                w = __variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            __variable_summaries(w)
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            __variable_summaries(bias)
        with tf.name_scope('layer_deconv2d'):
            deconv = tf.nn.conv2d_transpose(x, w, output_shape, stride, padding)
            #conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(deconv, bias)

        return out
    
def deconv2d(name, x, output_shape, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):

    with tf.variable_scope(name) as scope:
        deconv_o_b = __deconv2d_p(scope, x=x, output_shape=output_shape, w=w,  num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                              padding=padding,
                              initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            deconv_o_b = tf.layers.batch_normalization(deconv_o_b, training=is_training)
            if not activation:
                conv_a = deconv_o_b
            else:
                conv_a = activation(deconv_o_b)
        else:
            if not activation:
                conv_a = deconv_o_b
            else:
                conv_a = activation(deconv_o_b)

        def dropout_with_keep():
            return tf.nn.dropout(conv_a, dropout_keep_prob)/dropout_keep_prob

        def dropout_no_keep():
            return tf.nn.dropout(conv_a, 1.0)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.cond(is_training, dropout_with_keep, dropout_no_keep)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(conv_o_dr)

    return conv_o



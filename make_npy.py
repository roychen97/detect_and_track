import numpy as np
import tensorflow as tf
from utility import anchor_manipulator
from easydict import EasyDict as edict
import json

def make_npy(filen_name, config_args):

    out_shape = [config_args.input_size] * 2
    anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
        layers_shapes = config_args.layers_shapes,
        anchor_scales = config_args.anchor_scales,
        extra_anchor_scales = config_args.extra_anchor_scales,
        anchor_ratios = config_args.anchor_ratios,
        layer_steps = config_args.layer_steps)


    all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
    priorbox_data = anchor_creator.get_anchor_boxes(all_anchors, all_num_anchors_depth, all_num_anchors_spatial, 'priorbox_data')

    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        priorbox_data = tf.identity(priorbox_data, name = 'priorbox_data')

        priorbox_data = sess.run([priorbox_data]) 
        sess.close()
    print('priorbox_data = \n{}'.format(priorbox_data[0].shape)) 
    np.save(filen_name, priorbox_data[0])    
    # # =======
    # # priorbox = np.transpose(np.reshape(priorbox_data[0],[4,-1]),[1,0])
    # priorbox = np.reshape(priorbox_data[0],[-1,4])

    # print('xyxy prior_box = \n{}'.format(priorbox)) 


    # anchor_w = priorbox[:,2] - priorbox[:,0]
    # anchor_h = priorbox[:,3] - priorbox[:,1]
    # anchor_cx = (priorbox[:,2] + priorbox[:,0])/2
    # anchor_cy = (priorbox[:,3] + priorbox[:,1])/2

    # anchors = np.transpose(np.reshape(np.concatenate([ anchor_cx, anchor_cy, anchor_w, anchor_h],
    #         -1), [4, -1]),[1,0])
    # print('xywh anchors = \n{}'.format(anchors))


if __name__ == '__main__':
    config_name = 'config/mobilenetv2_300_person_wework_anchor2.json'
    # config_name = 'config/test_config.json'
    with open(config_name, 'r') as config_file:
        config_args_dict = json.load(config_file)
        config_args = edict(config_args_dict)
    make_npy('./priorbox_data.npy', config_args)


# out_shape = [96] * 2
# anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
#     layers_shapes = [(12, 12), (6, 6), (3, 3), (1, 1)],
#     anchor_scales = [(0.15,), (0.3,), (0.45,), (0.6,)],
#     extra_anchor_scales = [(0.2,), (0.4,), (0.5,), (0.8,)],
#     anchor_ratios = [(1.,), (1.,),(1.,),(1.,)],
#     layer_steps = [8, 16, 32, 96],
#     prior_scaling=[0.1, 0.1, 0.2, 0.2])


# SIZE = 300
# from net import ssd_mobilenetV2_300 as ssd_net
# LOGDIR = './logs_mobilenet300_people/'
# train_tfrecord_dir = './dataset/pascal_person_tfrecords'
# layers_shapes = [(19, 19), (10, 10), (5, 5), (3, 3), (2, 2), (1, 1)]
# anchor_scales = [(0.1,), (0.2,), (0.375,), (0.55,), (0.725,), (0.9,)]
# extra_anchor_scales = [(0.1414,), (0.2739,), (0.4541,), (0.6315,), (0.8078,), (0.9836,)]
# anchor_ratios = [(1., 2., .5), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., 3., .5, 0.3333), (1., 2., .5), (1., 2., .5)]
# layer_steps = [16, 32, 64, 100, 150, 300]
# layer_num = 6
# out_shape = [SIZE] * 2
# anchor_creator = anchor_manipulator.AnchorCreator(out_shape,
#     layers_shapes = layers_shapes,
#     anchor_scales = anchor_scales,
#     extra_anchor_scales = extra_anchor_scales,
#     anchor_ratios = anchor_ratios,
#     layer_steps = layer_steps)


# all_anchors, all_num_anchors_depth, all_num_anchors_spatial = anchor_creator.get_all_anchors()
# priorbox_data = anchor_creator.get_anchor_boxes(all_anchors, all_num_anchors_depth, all_num_anchors_spatial, 'priorbox_data')

# gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False)) as sess:
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     priorbox_data = tf.identity(priorbox_data, name = 'priorbox_data')

#     priorbox_data = sess.run([priorbox_data]) 
# print('priorbox_data = {}'.format(priorbox_data[0].shape)) 
# np.save('./priorbox96_data.npy', priorbox_data[0])    
import os
import tensorflow as tf
import numpy as np
import cv2 as cv
import time 
import math
# from ssdface_freeze import anchor_manipulator_np
from utility import draw_toolbox
from easydict import EasyDict as edict
import json

# config_name = 'config/mobilenetv2_96_face.json'
config_name = 'config/mobilenetv2_300_person.json'
with open(config_name, 'r') as config_file:
    config_args_dict = json.load(config_file)
    config_args = edict(config_args_dict)
model = 'ssdface_freeze/' + config_args.freeze_file + '.pb'

tf.app.flags.DEFINE_integer(     'num_classes', config_args.num_classes, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_integer(    'train_image_size', config_args.input_size,    'The size of the input image for the model to use.')
tf.app.flags.DEFINE_string(  'data_format', config_args.data_format,' channels_first or channels_last')
tf.app.flags.DEFINE_float(    'select_threshold', 0.2, 'Class-specific confidence score threshold for selecting a box.')
tf.app.flags.DEFINE_float(    'min_size', 0.1, 'The min size of bboxes to keep.')
tf.app.flags.DEFINE_float(    'nms_threshold', 0.45, 'Matching threshold in NMS algorithm.')
tf.app.flags.DEFINE_integer(    'nms_topk', 20, 'Number of total object to keep after NMS.')
tf.app.flags.DEFINE_integer(    'keep_topk', 200, 'Number of total object to keep for each image before nms.')
FLAGS = tf.app.flags.FLAGS


def preprocessing(np_image, input_format, output_format, data_format, bias=[128,128,128]):
    _R_MEAN = bias[0]
    _G_MEAN = bias[1]
    _B_MEAN = bias[2]
    image = cv.resize(np_image, (config_args.input_size, config_args.input_size), interpolation=cv.INTER_NEAREST)
    image = image.astype(float)
    
    if input_format == 'bgr':
      test_image = np.concatenate([np.expand_dims(image[:,:,2], axis = -1),
                                  np.expand_dims(image[:,:,1], axis = -1), 
                                  np.expand_dims(image[:,:,0], axis = -1)], axis = 2)
    else:
      test_image = image                                          
    test_image = test_image - np.expand_dims(np.expand_dims([_R_MEAN, _G_MEAN, _B_MEAN], axis = 0), axis = 0)
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

def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections + 0.00000001
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

def non_max_suppression(boxes, scores, nms_topk, threshold):	
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    
    if len(boxes_keep_index) > nms_topk:
      boxes_keep_index = boxes_keep_index[:nms_topk]
    return np.array(boxes_keep_index)




#### face detection ####
class ROTATESSD(object):
    def __init__(self):
        self.PATH_TO_CKPT = model
        self.PATH_TO_LABELS = '../models/face_label_map.pbtxt'
        self.NUM_CLASSES = config_args.num_classes
        self.sess = self._init_sess()
        self.fa_input = self.sess.graph.get_tensor_by_name('face/input_image:0')
        self.fa_priorbox_data = self.sess.graph.get_tensor_by_name('face/priorbox_data:0')    
        self.fa_conf_data = self.sess.graph.get_tensor_by_name('face/conf_data:0')    
        self.fa_loc_data = self.sess.graph.get_tensor_by_name('face/loc_data:0')
        self.data_format = 'channels_first'
        print('\n\n <<< FACE __init__ DONE >>>')
        tf.summary.FileWriter('./freezelogs', self.sess.graph)
        ########################
        # create input
        ########################


    def _init_sess(self):
        fa_graph = tf.Graph()
        with fa_graph.as_default() as graph:
            with tf.gfile.GFile(self.PATH_TO_CKPT, "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def,  name="face") 
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
            config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
            x = fa_graph.get_tensor_by_name('face/input_image:0')
            y = fa_graph.get_tensor_by_name('face/priorbox_data:0')
            node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            for node in node_names:
                print(node)
            return tf.Session(config=config)

    
    # def preprocessing(self, np_image, input_format, output_format, bias=[128,128,128]):
    #     #print('\n==\n input image = {}'.format(np_image[0,0,:]))
    #     _R_MEAN = bias[0]
    #     _G_MEAN = bias[1]
    #     _B_MEAN = bias[2]
    #     image = np_image.astype(float)
    #     image = cv.resize(image, (config_args.input_size, config_args.input_size))
    #     if input_format == 'bgr':
    #       test_image = np.concatenate([np.expand_dims(image[:,:,2], axis = -1),
    #                                   np.expand_dims(image[:,:,1], axis = -1), 
    #                                   np.expand_dims(image[:,:,0], axis = -1)], axis = 2)
    #     else:
    #       test_image = image                                          
    #     test_image = test_image - np.expand_dims(np.expand_dims([_R_MEAN, _G_MEAN, _B_MEAN], axis = 0), axis = 0)
    #     if output_format == 'bgr':
    #       output_image = np.concatenate([np.expand_dims(test_image[:,:,2], axis = -1),
    #                                     np.expand_dims(test_image[:,:,1], axis = -1), 
    #                                     np.expand_dims(test_image[:,:,0], axis = -1)], axis = 2)
    #     else:
    #       output_image = test_image   
    #     return np.expand_dims(output_image, axis = 0)


    #def proces_box(self, boxes, classes, scores, category_index):
    def groupes_boxes_and_labels(self,
        h,
        w,
        boxes,
        classes,
        scores,
        category_index,
        max_boxes_to_draw=20,
        min_score_thresh=.5):

      if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
      all_box = []
      for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            instance = dict()
            box = boxes[i]
            instance['box'] = list(map(int, [box[1]*w, box[0]*h, box[3]*w, box[2]*h]))
            if classes[i] in category_index.keys():
              class_name = category_index[classes[i]]['name']
            else:
              class_name = 'N/A'
            display_str = str(class_name)
            instance['class_name'] = class_name
            instance['scores'] = scores[i]
            all_box.append(instance)
      return all_box        

    def detect_nms(self, image):
        test_image = preprocessing(image, input_format = 'bgr', output_format = 'rgb', data_format = self.data_format, bias = config_args.bias)
        # test_image = np.ones((1,3,300,300)) * 0.5
        priorbox, conf_data,  loc_data = self.sess.run([self.fa_priorbox_data, self.fa_conf_data, self.fa_loc_data], feed_dict={self.fa_input: test_image})     
        priorbox = np.reshape(priorbox,[2,-1,4])
        loc_data = np.reshape(loc_data,[1,-1,4])
        conf_data = np.reshape(conf_data,[-1,2])
        anchor_w = priorbox[0,:,2] - priorbox[0,:,0]
        anchor_h = priorbox[0,:,3] - priorbox[0,:,1]
        anchor_cx = (priorbox[0,:,2] + priorbox[0,:,0])/2
        anchor_cy = (priorbox[0,:,3] + priorbox[0,:,1])/2
        # anchors = np.transpose(np.reshape(np.concatenate([ anchor_cx, anchor_cy, anchor_w, anchor_h],
        #     -1), [4, -1]),[1,0])
        # print('anchors = {}'.format(anchors[-50:,:]))
        bbox_cx = priorbox[1,:,0] * loc_data[0,:,1] * anchor_w + anchor_cx
        bbox_cy = priorbox[1,:,1] * loc_data[0,:,0] * anchor_h + anchor_cy
        bbox_w = np.exp(priorbox[1,:,2] * loc_data[0,:,3] ) * anchor_w
        bbox_h = np.exp(priorbox[1,:,3] * loc_data[0,:,2] ) * anchor_h
        bboxes_pred = np.transpose(np.reshape(np.concatenate([ bbox_cy - bbox_h/2, bbox_cx - bbox_w/2, bbox_cy + bbox_h/2, bbox_cx + bbox_w/2  ],
             -1), [4, -1]),[1,0])
        scores_pred = softmax(conf_data)[:,1]
        idxes = non_max_suppression(bboxes_pred, scores_pred, FLAGS.nms_topk, FLAGS.nms_threshold)
        return np.take(scores_pred, idxes), bboxes_pred[idxes,:]

    def detect_all(self, image):
        test_image = preprocessing(image, input_format = 'bgr', output_format = 'rgb', data_format = self.data_format)
        priorbox, conf_data,  loc_data = self.sess.run([self.fa_priorbox_data, self.fa_conf_data, self.fa_loc_data], feed_dict={self.fa_input: test_image})     
        all_scores = conf_data
        all_bboxes = loc_data
        all_labels = priorbox
        return all_labels, all_scores, all_bboxes

def read_labelme(json_filename, sub_folder, json_path):
    if not '.json' in json_filename:
        return False
    with open(os.path.join(json_path, sub_folder, json_filename)) as json_file:
        data = json.load(json_file)
    full_path = os.path.join(json_path, sub_folder, data["imagePath"])
    img_filename = full_path.split('/')[-1]
    # image = cv.imread(full_path)
    all_boxes = []
    if 'shapes' in data:
        for obj in data['shapes']:
            #if not 'person' in obj['class']:
            #    continue
            box = obj['points']
            box = [box[0][0], box[0][1],box[1][0],box[1][1]]
            box = list(map(int, box))
            all_boxes.append(box)
            # cv.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255,0,255), 2)            
    # cv.imshow('image', image)
    if len(all_boxes) > 0:
        img_name = json_filename.split('.')[0]
        txt = open(os.path.join('/home/scchiu/Workspace/Object-Detection-Metrics/ulsee_face_gt', '{}_{}.txt'.format(sub_folder, img_name) ), "w+") 
        for box in all_boxes:
            txt.write('person {} {} {} {}\n'.format(box[0],box[1],box[2],box[3]))
        txt.close()  

def main():
    face_detector = FACESSD()
    new_img = cv.imread('demo/test300.jpg')
    all_scores, all_bboxes = face_detector.detect_nms(new_img)
    for score, box in zip(all_scores, all_bboxes):
        if score < 0.5:
          continue
        x1 = int(box[1]*new_img.shape[1])
        y1 = int(box[0]*new_img.shape[0])
        x2 = int(box[3]*new_img.shape[1])
        y2 = int(box[2]*new_img.shape[0])
        cv.rectangle(new_img, (x1, y1), (x2, y2), (255,155,55), 2)
        cv.putText(new_img,  '{}'.format(int(score*1000)), (x1,y1),0,1,(200,255,255))
    cv.imshow('img', new_img)
    cv.waitKey()

                
if __name__ == '__main__':
    main()


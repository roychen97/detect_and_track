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
import cv2
import matplotlib.cm as mpcm
import numpy as np
import math
from dataset import dataset_common

def gain_translate_table():
    label2name_table = {}
    for class_name, labels_pair in dataset_common.ULSEE_LABELS.items():
        label2name_table[labels_pair[0]] = class_name
    return label2name_table

label2name_table = gain_translate_table()

def colors_subselect(colors, num_classes=21):
    dt = len(colors) // num_classes
    sub_colors = []
    for i in range(num_classes):
        color = colors[i*dt]
        if isinstance(color[0], float):
            sub_colors.append([int(c * 255) for c in color])
        else:
            sub_colors.append([c for c in color])
    return sub_colors

colors = colors_subselect(mpcm.plasma.colors, num_classes=21)
colors_tableau = [(255, 255, 255), (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

def center2point( center_y, center_x, height, width):
    # center_y, center_x, height, width = box
    # center_y = box[0]
    # center_x = box[1]
    # height = box[2]
    # width = box[3]
    angle = math.atan((center_y - 0.5)/( center_x - 0.5 + 1e-20))
    pi = math.pi
    angle = np.where(np.less_equal(center_x, 0.5) , pi - angle, -angle)
    angle = np.where(np.less_equal(angle, 0.0), pi + pi + angle, angle)
    
    rotation_matrix = np.stack([math.cos(angle), -math.sin(angle),  
                        math.sin(angle),math.cos(angle)], axis=0)
    rotation_matrix = np.reshape(rotation_matrix, (2, 2))
    height, width = width, height
    points = np.stack([[ -width / 2,  -height / 2], [ width / 2,  -height / 2], [ width / 2,  height / 2], [ -width / 2,  height / 2] ], axis=0)
    points = np.matmul(points, rotation_matrix) + [center_x, center_y]
    return points

def rotate_bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2, xy_order = 'xy', color=None):
    shape = img.shape
    scale = 0.8
    text_thickness = 3
    line_type = 8
    pointslist=[]
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        if scores[i] < 0.5: continue
        bbox = bboxes[i]
        points = center2point(bbox[0],bbox[1],bbox[2],bbox[3])
        points = (points * img.shape[0]).astype(int)
        pointslist.append(points)
        if color==None:
            color = colors_tableau[classes[i]]
        # Draw bounding boxes


        points = np.reshape(points.astype(int), [-1,1,2])
        cv2.polylines(img,[points],True,color,thickness)  

        y = int(bbox[0]*img.shape[0])
        x = int(bbox[1]*img.shape[0])
        # cv2.putText(img, '{}'.format(int(scores[i]*100)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)


        # Draw text
        # s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        # text_size is (width, height)
        # text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        # p1 = (p1[0] - text_size[1], p1[1])

        # cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        # cv2.putText(img, s, (points[0][0], points[0][1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img,pointslist

def rotate_bboxes_nodraw_on_img(img, classes, scores, bboxes, thickness=2, xy_order = 'xy', color=None):
    shape = img.shape
    scale = 0.8
    text_thickness = 3
    line_type = 8
    pointslist=[]
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        if scores[i] < 0.5: continue
        bbox = bboxes[i]
        points = center2point(bbox[0],bbox[1],bbox[2],bbox[3])
        points = (points * img.shape[0]).astype(int)
        pointslist.append(points)
        if color==None:
            color = colors_tableau[classes[i]]
        # Draw bounding boxes


        points = np.reshape(points.astype(int), [-1,1,2])
        #cv2.polylines(img,[points],True,color,thickness)  

        y = int(bbox[0]*img.shape[0])
        x = int(bbox[1]*img.shape[0])
        # cv2.putText(img, '{}'.format(int(scores[i]*100)), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)


        # Draw text
        # s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        # text_size is (width, height)
        # text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        # p1 = (p1[0] - text_size[1], p1[1])

        # cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        # cv2.putText(img, s, (points[0][0], points[0][1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img,pointslist
def bboxes_draw_on_img(img, classes, scores, bboxes, thickness=2, xy_order = 'xy'):
    shape = img.shape
    scale = 0.4
    text_thickness = 1
    line_type = 8
    for i in range(bboxes.shape[0]):
        if classes[i] < 1: continue
        bbox = bboxes[i]
        if xy_order != 'yx':
            bbox = [bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2]]
        color = colors_tableau[classes[i]]
        # Draw bounding boxes
        p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
        p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
        if (p2[0] - p1[0] < 1) or (p2[1] - p1[1] < 1):
            continue

        cv2.rectangle(img, p1[::-1], p2[::-1], color, thickness)
        # Draw text
        s = '%s/%.1f%%' % (label2name_table[classes[i]], scores[i]*100)
        # text_size is (width, height)
        text_size, baseline = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, text_thickness)
        p1 = (p1[0] - text_size[1], p1[1])

        cv2.rectangle(img, (p1[1] - thickness//2, p1[0] - thickness - baseline), (p1[1] + text_size[0], p1[0] + text_size[1]), color, -1)

        cv2.putText(img, s, (p1[1], p1[0] + baseline), cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), text_thickness, line_type)

    return img


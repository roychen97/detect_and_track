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

from datetime import datetime
import os
import random
import sys
import threading
import xml.etree.ElementTree as xml_tree

import numpy as np
import six
import tensorflow as tf

import dataset_common



def read_annotation(base_path, annotation):
    all_records = [] 
    print('======== path = {}'.format(annotation))   
    if 'train' in annotation:
      f = open('/home/scchiu/Data/WIDER_FACE/wider_face_split/wider_face_train_bbx_gt.txt', 'r')    
    else:
      f = open('/home/scchiu/Data/WIDER_FACE/wider_face_split/wider_face_val_bbx_gt.txt', 'r')    
    cnt = 0

    while True:
      cnt +=1
      line = f.readline().strip('\n')
      if not line:
          break
      if not 'jpg' in line:
          break

      # open image
      filename = line
      print(filename)
      img_data = dict()
      
      img_path = os.path.join(base_path, filename)
      img_data['img_path'] = img_path
      if not os.path.exists(img_path):
          n_windows = int(f.readline().strip('\n'))
          for i in range(n_windows):
              f.readline().strip('\n')
          if n_windows == 0:
              f.readline().strip('\n')
          continue

      # face number
      n_windows = int(f.readline().strip('\n'))
      img_data['n_windows'] = n_windows
      if n_windows == 0:
          f.readline().strip('\n')
          continue
      items = []


      for i in range(n_windows):
        obj = dict()
        cur_window = f.readline().strip('\n').split(' ')
        _xmin = float(cur_window[0])
        _ymin = float(cur_window[1])
        _xmax = _xmin + float(cur_window[2])  # ymin + width
        _ymax = _ymin + float(cur_window[3])  # xmin + height

        if (_ymax > _ymin) and (_xmax > _xmin): # valid point set
          obj['xmin'] = _xmin
          obj['ymin'] = _ymin
          obj['xmax'] = _xmax
          obj['ymax'] = _ymax
          items.append(obj)
          
          #classes_text.append('face'.encode('utf8'))
      img_data['faces'] = items
      all_records.append(img_data)
    f.close()
    print("read {} images".format(cnt))
    return all_records


if __name__ == '__main__':
  #all_records = read_annotation( '/home/scchiu/Data/WIDER_FACE/', 'train')
  all_records = read_annotation( '/home/scchiu/Data/WIDER_FACE/', 'val')

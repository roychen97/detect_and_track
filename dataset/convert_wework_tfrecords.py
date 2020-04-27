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
import xml.etree.ElementTree as ET

import numpy as np
import six
import tensorflow as tf
import cv2 as cv
import dataset_common
import json
import math

from utility import rotated_box

tf.app.flags.DEFINE_string('output_directory', 'dataset/wework_tfrecords__123',
                           'Output data directory')
tf.app.flags.DEFINE_integer('train_shards', 4,
                            'Number of shards in training TFRecord files.')
tf.app.flags.DEFINE_integer('validation_shards', 4,
                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('num_threads', 4,
                            'Number of threads to preprocess the images.')


RANDOM_SEED = 180428

FLAGS = tf.app.flags.FLAGS

if not os.path.exists(FLAGS.output_directory):
    os.makedirs(FLAGS.output_directory)

def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_list_feature(value):
    """Wrapper for inserting a list of bytes features into Example proto.
    """
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  if isinstance(value, six.string_types):
    value = six.binary_type(value, encoding='utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def read_annotation_roLabelImg(anno_path, img_path):
  all_records = []
  xml_list = os.listdir(anno_path)
  for idx, xml_filename in enumerate(xml_list):
    if not '.xml' in xml_filename:
      continue  
    tree = ET.parse(os.path.join(anno_path, xml_filename))
    root = tree.getroot()
    filename =  root.find('filename').text
    width = float(root.find('size').find('width').text)
    height = float(root.find('size').find('height').text)
    img_data = dict()
    items = []
    img_data['img_path'] = img_path
    img_data['img_name'] = filename + '.jpg'

    
    if idx < 3:
      full_path = os.path.join(img_path, img_data['img_name'])
      image = cv.imread(full_path)

    for member in root.iter('object'):
        if member[0].text=="bndbox":
            x0 = float(member[5][0].text)
            y0 = float(member[5][1].text)
            x1 = float(member[5][2].text)
            y1 = float(member[5][3].text)
            x = (x1 + x0) / 2
            y = (y0 + y1) / 2
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            a = 0.0
        elif member[0].text=="robndbox":
            x = (float(member[5][0].text))
            y = (float(member[5][1].text))
            w = (float(member[5][2].text))
            h = (float(member[5][3].text))
            a = (float(member[5][4].text))

        cx = x/width
        cy = y/height
        angle = rotated_box.center_to_angle(cy, cx, direction="CW")
        if not abs(abs(angle - a) / math.pi - round(abs(angle - a) / math.pi)) < 0.1:
            a = angle
            w, h = h, w
        
        obj = dict()
        obj['cx'] = cx
        obj['cy'] = cy
        obj['width'] = h/height
        obj['height'] = w/width
        items.append(obj)

        if idx < 3:
          pts = rotated_box.center_to_points(y/height, x/width, obj['height'], obj['width'], is_h_larger=True)
          pts = np.multiply(pts, width).astype(int)
          pts = np.reshape(pts.astype(int), [-1,1,2])
          cv.polylines(image,[pts],True,(0,255,255),2)  

    if idx < 3:
      cv.imshow(img_data['img_name'] , cv.resize(image, (1024,1024)))
    if len(items) > 0:
      img_data['objs'] = items
      all_records.append(img_data)
    # cv.imwrite(os.path.join('/home/scchiu/Pictures/tmp', img_data['img_name']), image)
  return all_records

def read_annotation_labelme_wework(anno_path, img_path, data_type):
  print('read label data {}'.format(anno_path))
  all_records = [] 
  test_json_list = os.listdir(anno_path)

  for idx, json_filename in enumerate(test_json_list):
    if not '.json' in json_filename:
      continue

    with open(os.path.join(anno_path, json_filename)) as json_file:
      json_data = json.load(json_file)
      img_data = dict()
      full_path = os.path.join(img_path, json_data["imagePath"])
      img_data['img_path'] = img_path
      img_data['img_name'] = json_data["imagePath"]
      height = float(json_data["imageHeight"])
      width = float(json_data["imageWidth"])
      items = []
      image = cv.imread(full_path)
      if 'shapes' in json_data:
        for obj in json_data['shapes']:
          if obj['shape_type'] == "rectangle":
            
            box = obj['points']
            box = [box[0][0], box[0][1],box[1][0],box[1][1]]
            if (box[0] > box[2]):
                box[0], box[2] = box[2], box[0]
            if (box[1] > box[3]):
                box[1], box[3] = box[3], box[1]
            obj['cx'] = (box[0] + box[2])/2
            obj['cy'] = (box[1] + box[3])/2
            w = box[2] - box[0]
            h = box[3] - box[1]
            if w > h:
              w, h = h, w
            obj['width'] = w
            obj['height'] = h
            items.append(obj)

          if obj['shape_type'] == "polygon":
            box = obj['points']
            if len(box) != 4:
              continue
            box = np.array(box)
            box = np.divide(box, [[width, height]])
            # cx, cy, w, h = get_size(box)
            cx, cy, w, h = rotated_box.point_to_center(box)
            
            obj['cx'] = cx
            obj['cy'] = cy
            obj['width'] = w
            obj['height'] = h
            if w == 0 or h == 0:
              print(img_data['img_name'], box)
              continue
            items.append(obj)

          # pts = center2point(cy, cx, h, w)
          pts = rotated_box.center_to_points(cy, cx, h, w, is_h_larger=True)
          pts = np.multiply(pts, width).astype(int)
          pts = np.reshape(pts.astype(int), [-1,1,2])
          cv.polylines(image,[pts],True,(0,255,255),5)          


        if len(items) > 0:
          # print('people num = {}'.format(len(items)))
          img_data['objs'] = items
          all_records.append(img_data)
          # print(img_data['objs'])
        # cv.imwrite(os.path.join('/home/scchiu/Pictures/tmp',img_data['img_name']), image)
        if idx < 3:
          cv.imshow(img_data['img_name'] , cv.resize(image,(1024,1024)))
  return all_records


def read_annotation_habbof(anno_path, img_path, data_type):
  print('read label data {}'.format(anno_path))
  all_records = [] 
  anno_list = os.listdir(anno_path)
  pi = math.pi
  im_size = 2048
  for idx, anno_filename in enumerate(anno_list):
    if not '.txt' in anno_filename:
      continue

    img_data = dict()
    img_data['img_path'] = img_path
    img_data['img_name'] = anno_filename.split('.')[0] + '.jpg'

    objects = np.loadtxt(os.path.join(anno_path,anno_filename), dtype=str)
    print(os.path.join(img_path,img_data['img_name']))
    image = cv.imread(os.path.join(img_path,img_data['img_name']))
    items = []  
    for obj in objects:
        box = dict() 
        # Get box properties
        cx = float(obj[1])/im_size
        cy = float(obj[2])/im_size
        w = float(obj[3])/im_size
        h = float(obj[4])/im_size
        angle = float(obj[5])*np.pi/180

        center_angle = math.atan((cy - 0.5)/(cx - 0.5 + 1e-10))
        center_angle = np.where(np.less_equal(cx, 0.5) , pi - center_angle, -center_angle)
        center_angle = np.where(np.less_equal(center_angle, 0.0), pi + pi + center_angle, center_angle)
        if abs(center_angle + angle) < pi/15 or abs(center_angle + angle -pi) < pi/15 or abs(center_angle + angle  - 2 * pi) < pi/15 :
            w, h = h, w
        box['cx'] = cx
        box['cy'] = cy
        box['width'] = w
        box['height'] = h
        items.append(box)
        if w == 0 or h == 0:
          print(img_data['img_name'] )
          # input()
        # pts = center2point(cy, cx, h, w)
        pts = rotated_box.center_to_points(cy, cx, h, w, is_h_larger=True)
        pts = np.multiply(pts, im_size).astype(int)
        pts = np.reshape(pts.astype(int), [-1,1,2])
        cv.polylines(image,[pts],True,(0,255,255),3)    

    if len(items) > 0:
      # print('people num = {}'.format(len(items)))
      img_data['objs'] = items
      all_records.append(img_data)
    
    if idx < 3:
      cv.imshow(img_data['img_name'] , cv.resize(image,(1024,1024)))
    # cv.imwrite(os.path.join('/home/scchiu/Pictures/tmp',img_data['img_name']), image)

  return all_records


def _convert_to_example(filepath, image_name, image_buffer, bboxes, labels, labels_text,
                        difficult, truncated, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    bboxes: List of bounding boxes for each image
    labels: List of labels for bounding box
    labels_text: List of labels' name for bounding box
    difficult: List of ints indicate the difficulty of that bounding box
    truncated: List of ints indicate the truncation of that bounding box
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  cx = []
  cy = []
  w = []
  h = []
  for b in bboxes:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([cx, cy, w, h], b)]
    # pylint: enable=expression-not-assigned
  channels = 3
  image_format = 'JPEG'
  example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': _int64_feature(height),
            'image/width': _int64_feature(width),
            'image/channels': _int64_feature(channels),
            'image/shape': _int64_feature([height, width, channels]),
            'image/object/bbox/cx': _float_feature(cx),
            'image/object/bbox/cy': _float_feature(cy),
            'image/object/bbox/width': _float_feature(w),
            'image/object/bbox/height': _float_feature(h),
            'image/object/bbox/label': _int64_feature(labels),
            'image/object/bbox/label_text': _bytes_list_feature(labels_text),
            'image/object/bbox/difficult': _int64_feature(difficult),
            'image/object/bbox/truncated': _int64_feature(truncated),
            'image/format': _bytes_feature(image_format),
            'image/filename': _bytes_feature(image_name.encode('utf8')),
            'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    # image = tf.image.resize(image, (600, 600))   
    return image


def _process_image(filename, coder):
  """Process a single image file.
  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  with tf.gfile.FastGFile(filename, 'rb') as f:
    image_data = f.read()

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _find_image_bounding_boxes(directory, cur_record, height, width):
  """Find the bounding boxes for a given image file.

  Args:
    directory: string; the path of all datas.
    cur_record: list of strings; the first of which is the sub-directory of cur_record, the second is the image filename.
  Returns:
    bboxes: List of bounding boxes for each image.
    labels: List of labels for bounding box.
    labels_text: List of labels' name for bounding box.
    difficult: List of ints indicate the difficulty of that bounding box.
    truncated: List of ints indicate the truncation of that bounding box.
  """

  # Find annotations.
  bboxes = []
  labels = []
  labels_text = []
  difficult = []
  truncated = []
  file_path = os.path.join(cur_record['img_path'],cur_record['img_name'])
  img = cv.imread(file_path)
  # print(cur_record['objs'])
  for box in cur_record['objs'] :
    
    label = 'person'
    labels.append(1)
    labels_text.append(label.encode('ascii'))
    difficult.append(0)
    truncated.append(0)
    # cv.rectangle(img, (int(box['cx']*600),int(box['cy']*600)), (int(box['width']*600),int(box['height']*600)), (255,255,0), 2)

    bboxes.append((float(box['cx'] ),
                  float(box['cy'] ),
                  float(box['width'] ),
                  float(box['height'] )
                  ))
  # if 'Frame63' in cur_record['img_name']:
  # cv.imwrite('/home/scchiu/Pictures/tmp/' + cur_record['img_name'], img)
  return bboxes, labels, labels_text, difficult, truncated

def _process_image_files_batch(coder, thread_index, ranges, name, directory, all_records, num_shards):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_directory, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      cur_record = all_records[i]
      filename = os.path.join(cur_record['img_path'], cur_record['img_name'])
      #filename = os.path.join(directory, cur_record[0], 'JPEGImages', cur_record[1])
      image_buffer, height, width = _process_image(filename, coder)
      bboxes, labels, labels_text, difficult, truncated = _find_image_bounding_boxes(directory, cur_record, height, width)
      #print('image_name = {}, ymin = {}, labels_text = {}'.format(filename, bboxes, labels_text))      
      example = _convert_to_example(cur_record['img_path'], cur_record['img_name'], image_buffer, bboxes, labels, labels_text,
                                    difficult, truncated, height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

def _process_image_files(name, directory, all_records, num_shards):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    directory: string; the path of all datas
    all_records: list of string tuples; the first of each tuple is the sub-directory of the record, the second is the image filename.
    num_shards: integer number of shards for this data set.
  """
  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  spacing = np.linspace(0, len(all_records), FLAGS.num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in range(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]])

  # Launch a thread for each batch.
  print('Launching %d threads for spacings: %s' % (FLAGS.num_threads, ranges))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  threads = []
  for thread_index in range(len(ranges)):
    args = (coder, thread_index, ranges, name, directory, all_records, num_shards)
    t = threading.Thread(target=_process_image_files_batch, args=args)
    t.start()
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(all_records)))
  sys.stdout.flush()

def _process_dataset(name, directory, all_splits, num_shards, data_type):
  # all_records is list of dict, format = {'img_path':'/path/to/image/folder', 
  # 'img_name': 'imagename.jpg', 
  # 'objs':[{'xmin':xmin,'ymin':ymin,'xmax':xmax,'ymax':ymax},
  # 	{'xmin':xmin,'ymisn':ymin,'xmax':xmax,'ymax':ymax}]}
  '''all_records = read_annotation_labelme('dataset/train_images', 
      'dataset/train_images', 
      data_type)
  '''
  all_records = []
  # all_records = all_records + read_annotation_labelme_wework('/home/scchiu/Data/WEWORK/Wework_Labeled', 
  #     '/home/scchiu/Data/WEWORK/Wework_Labeled', 
  #     data_type)

  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party1_Done', 
      '/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party1_Done')
  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party2_Done', 
      '/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party2_Done')
  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party3_Frame', 
      '/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party3_Frame')
  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party4_Frame', 
      '/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_Party4_Frame')
  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/2020407_wework_data', 
      '/home/scchiu/Data/WEWORK/2020407_wework_data')


  all_records = all_records + read_annotation_roLabelImg('/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_20200327_Frame_Labeled',
    '/home/scchiu/Data/WEWORK/Ceiling_OwnRecord_20200327_Frame_Labeled')

  habbof_folder = '/home/scchiu/Data/HABBOF'
  subfolders = os.listdir(habbof_folder)
  for subfolder in subfolders:
    if not  os.path.isdir(os.path.join(habbof_folder,subfolder)):
      continue
    all_records = all_records + read_annotation_habbof(os.path.join(habbof_folder,subfolder), 
        os.path.join(habbof_folder,subfolder), 
        data_type)
  cv.waitKey()


  print('read_annotation = {} records'.format(len(all_records)))
  shuffled_index = list(range(len(all_records)))
  random.seed(RANDOM_SEED)
  random.shuffle(shuffled_index)
  all_records = [all_records[i] for i in shuffled_index]

  _process_image_files(name, directory, all_records, num_shards)

def parse_comma_list(args):
    return [s.strip() for s in args.split(',')]

def main(unused_argv):
  assert not FLAGS.train_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with FLAGS.train_shards')
  assert not FLAGS.validation_shards % FLAGS.num_threads, (
      'Please make the FLAGS.num_threads commensurate with '
      'FLAGS.validation_shards')
  print('Saving results to %s' % FLAGS.output_directory)

  _process_dataset('train-wework', 
      'dataset', 
      'person', 
      FLAGS.train_shards, 
      data_type = 'train')



if __name__ == '__main__':
  tf.app.run()

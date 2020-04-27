import tensorflow as tf

graph_def_file = 'face_ssd_model_anchor_fz.pb'
input_arrays = ['input_image']
output_arrays = ['outputs']

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("face_ssd_model_anchor.tflite", "wb").write(tflite_model)
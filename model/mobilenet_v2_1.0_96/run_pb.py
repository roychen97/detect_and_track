import tensorflow as tf
import os

model = 'mobilenet_v2_1.0_96_frozen.pb'



mobilenet_graph = tf.Graph()
with mobilenet_graph.as_default() as graph:
    with tf.gfile.GFile(model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def,  name="") 
    gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    x = mobilenet_graph.get_tensor_by_name('input:0')
    y = mobilenet_graph.get_tensor_by_name('MobilenetV2/Predictions/Reshape_1:0')

    sess = tf.Session(config=config)
    tf.summary.FileWriter('./', sess.graph)
    node_names = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for node in node_names:
        print(node)

    all_vars = tf.trainable_variables()
    for v in all_vars:
        print("layer %s with shape " % (v.name))
        print(sess.run(tf.shape(v)))
        #print("%s with value %s" % (v.name, sess.run(v)))

    print("=========list of node=========")
    allname = [n.name for n in tf.get_default_graph().as_graph_def().node]
    for name in allname:
        print(name)

    print("===========all variables================")
    all_var = tf.global_variables()
    for var in all_var:
        print(var.name)


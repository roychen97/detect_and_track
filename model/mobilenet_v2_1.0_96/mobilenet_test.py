import tensorflow as tf
import os

#os.makedirs("/tmp/model")
#os.makedirs("/tmp/model-subset")

path =  "./"
meta =  path + "mobilenet_v2_1.0_96.ckpt.meta"

#model_folder = './weights/mobilenet_v1_1.0_224.ckpt'
model_folder = "./mobilenet_v2_1.0_96.ckpt"
clear_devices = True


#checkpoint = tf.train.get_checkpoint_state(model_folder)
#input_checkpoint = checkpoint.model_checkpoint_path




tf.reset_default_graph()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta)
    saver.restore(sess, tf.train.latest_checkpoint(path))
    sess.run(tf.global_variables_initializer())
    tf.summary.FileWriter('./', sess.graph)
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
'''
###### use tool ####
from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(file_name="../all_in_one/mtcnn-3000000",
                                      tensor_name=None, 
                                      all_tensors=False,
                                      all_tensor_names=True)

'''

'''
v1 = tf.Variable([0.1, 0.1], name="v1")
v2 = tf.Variable([0.2, 0.2], name="v2")


init_op = tf.global_variables_initializer()


saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init_op)
    #saver = tf.train.Saver({"my_v2": v2})
    ops = tf.assign(v2, [0.3, 0.3])
    sess.run(ops)

    print sess.run(tf.global_variables())

    save_path = saver.save(sess, "/tmp/model/model.ckpt")
'''

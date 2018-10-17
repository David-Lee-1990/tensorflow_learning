import tensorflow as tf
import json
import os
from tensorflow.python import pywrap_tensorflow

cwd = os.getcwd()


graph_path = '/Users/lee/tensorflow_learning/tf_save_restore/save/c/checkpoints_c/model-150.meta'
checkpoint_path = '/Users/lee/tensorflow_learning/tf_save_restore/save/c/checkpoints_c/model-150'

with tf.Session() as sess:
    
    """
    可以直接按如下命令导入meta图，实际是调用Saver类中的方法import_meta_graph，
    返回的是Saver实例
    """

    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess,checkpoint_path)

    variable_dic = {}
    for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        v = sess.run(variable)
        variable_dic[variable._shared_name] = str(v)
    with open("variable_value.json","w") as f:
        json.dump(variable_dic, f)
    with open("variable_name.txt","w") as f:
        for key in variable_dic.keys():
            f.write(key+"\n")





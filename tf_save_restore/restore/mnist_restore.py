import sys
sys.path.append("..")

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from save import input_data
import os

cwd = os.getcwd()

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

graph_path = '/Users/lee/tf_save_restore/save/checkpoints/model-100.meta'
checkpoint_path = '/Users/lee/tf_save_restore/save/checkpoints/model-100'

with tf.Session() as sess:
    
    """
    可以直接按如下命令导入meta图，实际是调用Saver类中的方法import_meta_graph，
    返回的是Saver实例
    """

    saver = tf.train.import_meta_graph(graph_path)
    saver.restore(sess,checkpoint_path)

    # print(tf.get_default_graph().as_graph_def())
    # with open('nodeDef.txt','w') as f:
    #     for n in tf.get_default_graph().as_graph_def().node:
    #         f.write(str(n))
    #         f.write('\n')
    #         print(n)

      

    # print(saver.as_saver_def())
    # print(saver.to_proto()) 

    """
    saverDef 格式如下：

    filename_tensor_name: "save/Const:0"
    save_tensor_name: "save/control_dependency:0"
    restore_op_name: "save/restore_all"
    max_to_keep: 10
    keep_checkpoint_every_n_hours: 10000.0
    version: V2

    """

    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    # with open('variable_shape.txt','w') as f:
    #     for key in sorted(var_to_shape_map):
    #         print("{}:{}".format(key,str(var_to_shape_map[key])))  
    #         f.write("{}:{}".format(key,str(var_to_shape_map[key]))) 
    #         f.write('\n')
    

    # with open('variables.txt','w') as f:
    #     for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #         print(variable)
    #         f.write(str(variable)+"\n")

    # with open('variables_value.txt','w') as f:
    #     for variable in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #         f.write(str(variable)+"\n")
    #         v = sess.run(variable)
    #         f.write(str(v)+'\n')

    # with open('trainable_variables.txt','w') as f:
    #     tvs = [v for v in tf.trainable_variables()]
    #     for v in tvs:
    #         f.write(v.name+'\n')
    #         f.write(str(sess.run(v)))
    #         f.write('\n')

    # with open('global_variables.txt','w') as f:
    #     tvs = [v for v in tf.global_variables()]
    #     for v in tvs:
    #         f.write(v.name+'\n')
    #         f.write(str(sess.run(v)))
    #         f.write('\n')

    graph =  tf.get_default_graph()

    # ops = [o for o in graph.get_operations()]
    # with open('operation_names.txt','w') as f:
    #     for o in ops:
    #         print(o.name)
    #         f.write(o.name+"\n")

    x = graph.get_tensor_by_name('Placeholder:0')
    y = graph.get_tensor_by_name('Placeholder_1:0')

    # train_step = graph.get_operation_by_name('LossPredict/train_step')
    # for i in range(100):
    #     print('transfer learning: %d'%(i))
    #     batch = mnist.train.next_batch(10)
    #     train_step.run(feed_dict={x:batch[0],y:batch[1]})
    #     if i%10 ==0:
    #         saver.save(sess,cwd+'/checkpoints/model',global_step=i)

    """

    基于存储的模型，增加新的操作

    """

    

    ml1 = graph.get_tensor_by_name('MLP_1/Relu:0') 
    # 这里要加 ：0，否则MLP_1/Relu是opration

    ml1 = tf.stop_gradient(ml1) 
    # 加这一句，是冻结ml1之前的层的所有系数，因为bp从ml1传不回去了

    ml1_shape = ml1.get_shape().as_list()


    with tf.variable_scope('new_1'):

        w_1 = tf.Variable(tf.truncated_normal([ml1_shape[1],20],stddev=0.5)                                                )
        bias_1 = tf.Variable(tf.constant(0.05,shape=[20]))
        ml2 = tf.nn.relu(tf.add(tf.matmul(ml1,w_1),bias_1),name='Relu')

    with tf.variable_scope('new_2'):

        w_2 = tf.Variable(tf.truncated_normal([20,10],stddev=0.5)                            )
        bias_2 = tf.Variable(tf.constant(0.05,shape=[10]))
        y_conv = tf.nn.softmax(tf.add(tf.matmul(ml2,w_2),bias_2),
                                name='outputs')

    with tf.variable_scope("LossPredict_new"):
        cross_entropy = -tf.reduce_mean(y*tf.log(y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                            name='train_step')

    tf.global_variables_initializer().run()

    for i in range(200):
        print('step:%d'%(i))
        batch = mnist.train.next_batch(10)
        train_step.run(feed_dict={x:batch[0],y:batch[1]})
        if i % 10 == 0:
            saver.save(sess,cwd+'/checkpoints/model',global_step=i)





    



    

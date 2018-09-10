import tensorflow as tf 
import input_data
import os

cwd_path = os.getcwd()

def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)

def conv2d(x,W,name):
    return tf.nn.conv2d(x,W,
                    strides=[1,1,1,1],
                    padding="SAME",
                    name=name)

def max_pool_2x2(x,name):
    return tf.nn.max_pool(x,
                        ksize=[1,2,2,1],
                        strides=[1,2,2,1],
                        padding="SAME",
                        name=name)

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
x = tf.placeholder(dtype=tf.float32, shape=[None,784])
y_ = tf.placeholder(dtype=tf.float32, shape=[None,10])

x_image = tf.reshape(x,[-1,28,28,1])
with tf.variable_scope('CONV_1'):

    W_conv1 = weight_variable([5,5,1,32],"W")
    b_conv1 = bias_variable([32],"b")
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1,"convolution_op")+b_conv1,name='Relu')
    h_pool1 = max_pool_2x2(h_conv1,name='Pool')

with tf.variable_scope('COV_2'):
    W_conv2 = weight_variable([5,5,32,64],"W")
    b_conv2 = bias_variable([64],"b")
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,'convolution_op')+b_conv2,name='Relu')
    h_pool2 = max_pool_2x2(h_conv2,name='Pool')

with tf.variable_scope("MLP_1"):
    W_fc1 = weight_variable([7*7*64,1024],"W_fc")
    b_fc1 = bias_variable([1024],"bias_fc")
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.add(tf.matmul(h_pool2_flat,W_fc1),b_fc1),'Relu')

with tf.variable_scope("MLP_2"):
    W_fc2 = weight_variable([1024,10],"W_fc")
    b_fc2 = bias_variable([10],"bias_fc")
    y_conv = tf.nn.softmax(tf.add(tf.matmul(h_fc1,W_fc2),b_fc2),name='outputs')

with tf.variable_scope("LossPredict"):
    cross_entropy = -tf.reduce_mean(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy,
                                                        name='train_step')
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"),name='accuracy')
    tf.summary.scalar('Training accuracy',accuracy)

merge_summary = tf.summary.merge_all()

with tf.Session() as sess:
    summary = tf.summary.FileWriter(cwd_path+'/summary')
    summary.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    print('=============Training===============')
    for i in range(200):
        print('step:%d'%(i))
        batch = mnist.train.next_batch(10)
        train_step.run(feed_dict={x:batch[0],y_:batch[1]})
        train_summary = sess.run(merge_summary,feed_dict={x:batch[0],y_:batch[1]})
        summary.add_summary(train_summary,i)
        if i % 10 == 0:
            saver.save(sess,cwd_path+'/checkpoints/model',global_step=i)
    


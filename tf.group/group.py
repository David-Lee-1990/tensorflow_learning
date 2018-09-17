import tensorflow as tf 

w_1 = tf.Variable(tf.random_normal([1],mean=0,stddev=1),"w_1")
w_2 = tf.Variable(tf.random_normal([1],mean=0,stddev=1),"w_2")
a_1 = tf.assign(w_1,[1])
a_2 = tf.assign(w_2,[0.5])
group_0 = tf.group(a_1,a_2) # 批量操作打包执行，节省代码量

mul = tf.multiply(w_1,2,"multiply")
add = tf.add(mul,2,"add")
group_1 = tf.group(mul,add)

tuple = tf.tuple([mul,add])

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(w_1.eval(),w_2.eval()) # [-3.1661828] [-0.5261834]
	print(sess.run(group_0)) # None
	print(w_1.eval(),w_2.eval()) # [1.] [0.5]
	print(sess.run(tuple)) # [2,4]

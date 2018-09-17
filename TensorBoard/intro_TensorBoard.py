import tensorflow as tf

# 定义简单加法计算图
input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
input2 = tf.Variable(tf.random_uniform(shape=[3]), name='input2')
output = tf.add_n([input1, input2], name='add')

# 生成一个写日志的writer，并且将当前的TensorFlow计算图写入日志，TensorFlow提供多种写日志文件的API
writer = tf.summary.FileWriter('./log', tf.get_default_graph())
writer.close()


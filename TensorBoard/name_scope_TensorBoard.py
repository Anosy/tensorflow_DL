import tensorflow as tf

# 将输入定义放入各自的命名空间中，从而使得TensorBoard可以根据命名空间来整理可视化效果图上的结果
with tf.name_scope('input1'):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')

with tf.name_scope('input2'):
    input2 = tf.Variable(tf.random_uniform([3]), name='input2')

output = tf.add_n([input1, input2], name='add')

writer = tf.summary.FileWriter(logdir='./log', graph=tf.get_default_graph())
writer.close()
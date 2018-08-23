import tensorflow as tf
import numpy as np

# 配置网络参数
input_node = 784
output = 10
image_size = 28
num_channels = 1  # 使用的是Mnist手写字体，只有黑白的单一像素
num_labels = 10

# 第一层卷积的尺寸和参数
conv1_deep = 32
conv1_size = 5
# 第二层卷积层的尺寸和深度
conv2_deep = 64
conv2_size = 5
# 全连接层的节点个数
fc_size = 512


# 定义卷积神经网络的前向传播的过程，这里添加上了dropout来防止过拟合
def inference(input_tensor, train, regularizer):

    # 第一层卷积的过程,输入的大小为28*28*1，输出为28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weight = tf.get_variable('weight', [conv1_size, conv1_size, num_channels, conv1_deep], initializer=
        tf.contrib.layers.xavier_initializer())
        conv1_biases = tf.get_variable('bias', [conv1_deep], initializer=tf.constant_initializer(0.0))

        conv1 = tf.nn.conv2d(input=input_tensor, filter=conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层最大池化，输入为28*28*32，输出为14*14*32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第三层卷积，输入为14*14*32，输出为14*14*64
    with tf.variable_scope('layer3-conv2'):
        conv2_weight = tf.get_variable('weight', [conv2_size, conv2_size, conv1_deep, conv2_deep], initializer=
        tf.contrib.layers.xavier_initializer())
        conv2_biases = tf.get_variable('bias', [conv2_deep], initializer=tf.constant_initializer(0.0))

        conv2 = tf.nn.conv2d(input=pool1, filter=conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 第四层最大池化，输入为14*14*64，输出为7*7*64
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # 第五层全连接层，但是这一层需要先将该层的矩阵给拉直成一个向量, 而且还加入了dropout和l2正则化，输入为7*7*64，输出为512
    pool2_shape = pool2.get_shape().as_list()  # shape的形式为[batch_size, 7, 7, 64]
    nodes = pool2_shape[1] * pool2_shape[2] * pool2_shape[3]
    reshaped = tf.reshape(pool2, [pool2_shape[0], nodes])   # 转化为batch_size*3136

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, fc_size], initializer=tf.contrib.layers.xavier_initializer())
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [fc_size], initializer=tf.constant_initializer(0.0))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:  # 只有训练过程才用到dropout
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层全连接层，输入为512，输出为10的向量
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable('weight', [fc_size, num_labels], initializer=tf.contrib.layers.xavier_initializer())
        if train != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable('bias', [num_labels], initializer=tf.constant_initializer(0.0))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit
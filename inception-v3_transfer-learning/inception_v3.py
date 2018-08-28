# slim库，是tensorflow中的轻量级的库。本程序主要是介绍slim库的使用，来搭建inception_v3网络的核心结构
import tensorflow.contrib.slim as slim
import tensorflow as tf

# arg_scope可以设置默认的参数取值，列表中的函数都是默认的参数取值
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):
    # net为上层输出的节点矩阵
    net = ...
    # 定义其中一个Inception模块，便且为其声明一个统一的变量命名空间
    with tf.variable_scope('Mixed_7c'):
        # 为每一个路径分支来声明一个命名空间

        # Inception 模块中第一条路径
        with tf.variable_scope('Branch_0'):
            # 卷积层的深度为320，1x1的卷积核
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')
        # Inception 模块中第二条路径
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            # tf.concat 函数可以将多个矩阵给拼接起来。tf.concat函数的第一个参数指定了拼接的维度，这里的'3'表示在深度上拼接
            # Inception_v3中将3x3卷积给改变成1x3和3x1卷积的结合，从而节约参数，
            branch_1 = tf.concat(3, [slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                                     slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')])
        # Inception 模块的第三条路径
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3, [slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                                     slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')])
        # Inception 模块的第四条路径
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

        # 模块的最后输出，将四个输出给合并
        net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])


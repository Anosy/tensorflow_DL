import os
import tensorflow as tf
import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import mnist_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 30001
MOVING_AVERAGE_DECAY = 0.99
# 模型保持的路径
MODEL_SAVE_PATH = './model_mnist/'
MODEL_NAME = 'model.ckpt'


def train(mnist):
    # 定义输入，输出
    x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')
    # 正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 考虑前向传播中添加上正则化，y中不体现. 这里使用了add_to_collection将其加入到了losses的集合中
    y = mnist_inference.inference(x, regularizer)
    # 当前的迭代次数
    global_step = tf.Variable(0, trainable=False)
    # 滑动平均
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    # 计算损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(
        tf.get_collection('losses'))  # 添加上正则化,tf.get_collection()表示返回名称位losses的列表, tf.add_n表示将列表中的元素相加并且返回
    # 学习率指数衰减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 训练步骤
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow 持久化类
    saver = tf.train.Saver()
    # 开启会话
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(1, TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})

            if i % 1000 == 0:
                print('After %d training steps, loss on training batch is %g' % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step)

if __name__ == '__main__':
    train(mnist)
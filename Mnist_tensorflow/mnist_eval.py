import tensorflow as tf
import time
import input_data
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import mnist_inference
import mnist_train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

EVAL_SHAPE = 5000
# 每10秒加载一次最新的模型，并且在测试数据上测试最新的模型的准确率
EAVL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:  # 创建一个新的计算图g
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [EVAL_SHAPE, mnist_inference.image_size, mnist_inference.image_size,
                                        mnist_inference.num_channels], name='x-input')
        y_ = tf.placeholder(tf.float32, [EVAL_SHAPE, mnist_inference.output], name='y-output')
        xs = np.reshape(mnist.validation.images, (
        EVAL_SHAPE, mnist_inference.image_size, mnist_inference.image_size, mnist_inference.num_channels))
        ys = mnist.validation.labels
        validate_feed = {x: xs, y_: ys}
        # 调用分装好的函数来计算前向传播，以为在测试的时候不需要关注正则化的问题，所以将regularizer=None
        y = mnist_inference.inference(x, None, None)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()

        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 获取目录中最新模型的文件名
            ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                print('After %s training steps validation accuracy = %g' % (global_step, accuracy_score))
            else:
                print('NO checkpoint file found')
                return time.sleep(EAVL_INTERVAL_SECS)


if __name__ == '__main__':
    evaluate(mnist)

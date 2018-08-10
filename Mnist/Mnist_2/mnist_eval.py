import tensorflow as tf
import time
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import mnist_inference
import mnist_train
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def evaluate(mnist):
    with tf.Graph().as_default() as g:  # 创建一个新的计算图g
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [None, mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, mnist_inference.OUTPUT_NODE], name='y-output')

        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        # 调用分装好的函数来计算前向传播，以为在测试的时候不需要关注正则化的问题，所以将regularizer=None
        y = mnist_inference.inference(x, None)
        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # 通过变量重命名的方式来加载模型,这里使得模型在加载参数的时候，只加载滑动平均后的参数
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


if __name__ == '__main__':
    evaluate(mnist)
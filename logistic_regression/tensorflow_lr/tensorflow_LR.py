import numpy as np
from lr_utils import load_dataset
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[None, 12288])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([12288, 1]))
b = tf.Variable(tf.zeros([1]))

y = tf.matmul(x, W) + b

# cost = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
# cosy = cost + tf.contrib.layers.l1_regularizer(0.1)(W) # 加上了l2正则化

correct_pred = tf.equal((tf.nn.sigmoid(y) > 0.5), (y_ > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

# 加载数据
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)
# 归一化数据
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.
train_set_y = train_set_y.T
test_set_y = test_set_y.T
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    STEPS = 1000
    for i in range(STEPS):
        _, c = sess.run([optimizer, cost], feed_dict={x: train_set_x, y_:train_set_y})

        if i % 100 == 0:
            print('After %d training step(s), cross entropy on all data is %s' % (i, c))
    acc_train = accuracy.eval(session=sess, feed_dict={x: train_set_x, y_:train_set_y})
    acc_test = accuracy.eval(session=sess, feed_dict={x: test_set_x, y_: test_set_y})
    print(("train accuracy: % 3.f%%" % (acc_train *100)))
    print(("test accuracy: % 3.f%%" % (acc_test*100)))
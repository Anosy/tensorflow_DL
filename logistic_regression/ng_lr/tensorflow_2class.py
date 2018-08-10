import tensorflow as tf
import numpy as np
from numpy.random import RandomState

# 定义训练集的batch_size
batch_size = 8
# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# 为输入定义占位符
x = tf.placeholder(tf.float32, shape=[None, 2], name='x-input')
y_ = tf.placeholder(tf.float32, shape=[None, 1], name='y-input')
# 前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
# 定义损失函数
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
# 反向传播
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 定义训练数据
rdm = RandomState(1)
data_size = 128
X = rdm.rand(data_size, 2)
Y = [[int(x1+x2 <1) for (x1, x2) in X]]
Y = np.array(Y).T

# 创建会话来运行程序
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % data_size
        end = min(start+batch_size, data_size)

        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x:X, y_:Y})
            print('After %d training step(s), cross entropy on all data is %g' % (i, total_cross_entropy))

    print(sess.run(w1))
    print(sess.run(w2))
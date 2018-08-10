import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # data = pd.read_csv('data.csv', header=None)
    # x, y = data[0], data[1]
    w = 2.0
    b = 1.0
    xs = np.linspace(-3, 3, 100)
    ys = xs*w + b + np.random.uniform(-1, 1,100) # 增加随机抖动

    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    W = tf.Variable(tf.random_normal([1]), name='weight')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(0.1, global_step, 20, 0.9, staircase=True)
    pred = tf.add(tf.multiply(X, W), b)

    cost = tf.square(Y-pred, name='cost')
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    n_sample = xs.shape

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            total_cost = 0.0
            for x, y in zip(xs, ys):
                _, c = sess.run([optimizer, cost], feed_dict={X:x, Y:y})
                total_cost += c
            if i % 10==0:
                print('Epoch {0}: {1}'.format(i, total_cost/n_sample))
        W, b = sess.run([W,b])
        print('W=\n',W)
        print('b=\n', b)
        plt.plot(xs, ys, 'bo', label='Real data')
        plt.plot(xs, xs * W + b, 'r', label='Predicted data')
        plt.legend()
        plt.show()
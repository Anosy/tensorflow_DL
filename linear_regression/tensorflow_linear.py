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

    pred = tf.add(tf.multiply(X, W), b)

    cost = tf.reduce_mean(tf.square(Y-pred))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    n_sample = xs.shape

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            total_cost = 0.0  # 每轮迭代后，下次迭代的总损耗清0
            for x, y in zip(xs, ys):
                _, c = sess.run([optimizer, cost], feed_dict={X:x, Y:y})
                print(c)
                total_cost += c  # 每轮迭代的总的损耗
            if i % 10==0:  # 每10次迭代打印损耗
                print('Epoch {0}: {1}'.format(i, total_cost/n_sample))
        W, b = sess.run([W,b])
        print('W=\n',W)
        print('b=\n', b)
        plt.plot(xs, ys, 'bo', label='Real data')
        plt.plot(xs, xs * W + b, 'r', label='Predicted data')
        plt.legend()
        plt.show()
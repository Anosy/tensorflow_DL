import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 定义学习率和输入的样本
learning_rate = 0.001
x_data = np.array([[0., 0.], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
y_data = np.array([[0], [1], [0], [1]])
# 定义占位符
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
# 定义权重
weights = {
    'w1': tf.Variable(tf.random_normal([2, 16])),
    'w2': tf.Variable(tf.random_normal([16, 1]))
}
# 定义偏执
biases = {
    'b1': tf.Variable(tf.zeros([16])),
    'b2': tf.Variable(tf.zeros([1]))
}
# 定义网络结构
def dnn(X, weights, biases):
    d1 = tf.matmul(x, weights['w1']) + biases['b1']
    d1 = tf.nn.relu(d1)
    d2 = tf.matmul(d1, weights['w2']) + biases['b2']
    # d2 = tf.nn.sigmoid(d2)
    return d2
# 预测
pred = dnn(x, weights, biases)
# 定义不同的损失函数
# cost = tf.reduce_mean(tf.square(y - pred)) # 均方差损失函数
# cost = -tf.reduce_mean(y * tf.log(tf.clip_by_value(pred, 1e-10, 1.0)) + (1-y)*tf.log(tf.clip_by_value(1-pred, 1e-10, 1.0))) # sigmoid交叉熵损失函数
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=y))  # sigmoid交叉熵损失函数
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  # softmax交叉熵损失函数,不适用于二分类问题
# 优化
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 计算精确度
correct_pred = tf.equal((pred > 0.5), (y > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# 初始化
init = tf.initialize_all_variables()
# 开启会话

with tf.Session() as sess:
    sess.run(init)
    for i in range(2500):
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        acc = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
        loss = sess.run(cost, feed_dict={x: x_data, y: y_data})
        if (i % 100 == 0):
            print("Step " + str(i) + "    loss " + "{:.6f}".format(loss))
            print("Step " + str(i) + "    acc " + "{:.6f}".format(acc))
            print('predict:\n', sess.run(pred, feed_dict={x: x_data}))
    print("Optimization Finished!")

    # 绘制图片
    xx, yy = np.mgrid[-0.1:1.1:.05, -0.1:1.1:.05]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(pred, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)

    plt.scatter(x_data[:, 0], x_data[:, 1], c=np.squeeze(y_data), cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
    plt.show()



import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def make_data():
    data= []
    label = []
    np.random.seed(0)

    for i in range(150):
        x1 = np.random.uniform(-1, 1)
        x2 = np.random.uniform(0, 2)
        if x1 ** 2 + x2 ** 2 <1:
            data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)]) # 最大随机性
            # data.append([x1, x2])
            label.append(0)
        else:
            data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
            # data.append([x1, x2])
            label.append(1)

    data = np.array(data).reshape(-1, 2)
    label = np.array(label).reshape(-1, 1)
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    # plt.show()
    return data, label

def get_weight(shape, lambda1):
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    return var


if __name__ == '__main__':
    data, label = make_data()  # 获取数据

    x = tf.placeholder(tf.float32, shape=(None, 2))
    y_ = tf.placeholder(tf.float32, shape=(None, 1))
    sample_size = len(data)
    # 每层的节点的个数
    layer_dimension = [2, 10, 5, 3, 1]
    # 神经网络的层数
    n_layers = len(layer_dimension)
    # 这个变量维护前向传播时最深层的节点，开始的时候就是输入层
    cur_layers = x
    # 当前层的节点个数
    in_dimension = layer_dimension[0]

    # 循环生成网络结构
    for i in  range(1, n_layers):
        out_dimension = layer_dimension[i]  # 下一层的节点数
        # 生成当前层的权重
        weight = get_weight([in_dimension, out_dimension], 0.003)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension])) # 偏置
        cur_layers = tf.nn.relu(tf.matmul(cur_layers, weight) + bias)
        in_dimension = layer_dimension[i]

    y = cur_layers

    mse_loss = tf.reduce_mean(tf.square(y - y_))
    tf.add_to_collection('losses', mse_loss)  # 将均方误差加入到losses中

    l2_loss = tf.add_n(tf.get_collection('losses'))

    # 1. 训练不带正则项的损失函数
    # train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(mse_loss)
    # TRAINING_STEPS = 40000

    # with tf.Session() as sess:
        # tf.global_variables_initializer().run()
        # for i in range(TRAINING_STEPS):
            # sess.run(train_op, feed_dict={x:data, y_:label})
            # if i % 2000 == 0:
                # print("After %d steps, mse_loss: %f" % (i, sess.run(mse_loss, feed_dict={x: data, y_: label})))

        # # 画出训练后的曲线
        # xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
        # grid = np.c_[xx.ravel(), yy.ravel()]
        # probs = sess.run(y, feed_dict={x: grid})
        # probs = probs.reshape(xx.shape)  # 将其给整形成原始输入点的样子

    # plt.scatter(data[:, 0], data[:, 1], c=label, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    # plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
    # plt.show()
	
	 # 2. 训练带正则项的损失函数
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(l2_loss)
    TRAINING_STEPS = 40000

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            sess.run(train_op, feed_dict={x:data, y_:label})
            if i % 2000 == 0:
                print("After %d steps, mse_loss: %f" % (i, sess.run(mse_loss, feed_dict={x: data, y_: label})))

        # 画出训练后的曲线
        xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)  # 将其给整形成原始输入点的样子

    plt.scatter(data[:, 0], data[:, 1], c=label, cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
    plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
    plt.show()


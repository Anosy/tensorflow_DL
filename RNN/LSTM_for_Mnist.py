import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 参数设置
BATCH_SIZE = 100  # BATCH的大小，相当于一次处理50个image
TIME_STEP = 28  # 一个LSTM中，输入序列的长度，image有28行
INPUT_SIZE = 28  # x_i 的向量长度，image有28列
LR = 0.01  # 学习率
NUM_UNITS = 100  # 多少个LTSM单元
ITERATIONS = 8000  # 迭代次数
N_CLASSES = 10  # 输出大小，0-9十个数字的概率

graph = tf.Graph()
with graph.as_default():
    # 定义placeholder
    train_x = tf.placeholder(tf.float32, shape=[None, TIME_STEP * INPUT_SIZE])
    image = tf.reshape(train_x, [-1, TIME_STEP, INPUT_SIZE])
    train_y = tf.placeholder(tf.float32, shape=[None, N_CLASSES])

    # 定义lstm结构
    rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)

    outputs, final_state = tf.nn.dynamic_rnn(
        cell=rnn_cell,
        inputs=image,
        initial_state=None,
        dtype=tf.float32,
        time_major=False
    )
    # print(outputs.shape)
    output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)  # 通过全连接层，得到输出的结果(batch_size, N_CLASSES)
    # print(output.shape)

    # 定义损失函数，优化方法和精确度
    # 损失函数
    loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=output)
    # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(train_y, 1), logits=output)
    # loss = tf.reduce_mean(loss)
    # 优化方法
    train_op = tf.train.AdamOptimizer(LR).minimize(loss)
    # 精确度
    correct_prediction = tf.equal(tf.argmax(train_y, axis=1), tf.argmax(output, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for step in range(ITERATIONS):
        x, y = mnist.train.next_batch(BATCH_SIZE)
        test_x, test_y = mnist.test.next_batch(5000)
        _, loss_ = sess.run([train_op, loss], feed_dict={train_x: x, train_y: y})
        if step % 500 == 0:
            acc = sess.run(accuracy, feed_dict={train_x: test_x, train_y: test_y})
            print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % acc)

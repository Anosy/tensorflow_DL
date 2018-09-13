import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt

TIME_STEPS = 10  # 时间步长
BATCH_SIZE = 128
HIDDEN_UNITS_1 = 20  # lstm模块每个时间步只有一个输出
HIDDEN_UNITS_2 = 1
LEARNING_RATE = 0.001
EPOCH = 50

TRAIN_EXAMPLES = 11000
TEST_EXAMPLES = 1100


# -------------------------------------------------生成数据-------------------------------------------------------------
# 按照一定的TIME_STEPS来生成数据
def generate(seq):
    X = []
    y = []
    for i in range(len(seq) - TIME_STEPS):
        X.append([seq[i:i + TIME_STEPS]])
        y.append([seq[i + TIME_STEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


seq_train = np.sin(np.linspace(start=0, stop=100, num=TRAIN_EXAMPLES, dtype=np.float32))
seq_test = np.sin(np.linspace(start=100, stop=110, num=TEST_EXAMPLES, dtype=np.float32))

# plt.plot(np.linspace(start=0, stop=100, num=TRAIN_EXAMPLES, dtype=np.float32), seq_train)
# plt.plot(np.linspace(start=100, stop=110, num=TEST_EXAMPLES, dtype=np.float32), seq_test)
# plt.show()

X_train, y_train = generate(seq_train)
# print(X_train.shape, y_train.shape)
X_test, y_test = generate(seq_test)

# reshape to （batch,time_steps,input_size）
X_train = np.reshape(X_train, newshape=(-1, TIME_STEPS, 1))
X_test = np.reshape(X_test, newshape=(-1, TIME_STEPS, 1))
# print(X_train.shape)
# print(X_test.shape)

# 绘制真实的测试集的结果
plt.plot(range(1000), y_test[:1000, 0], "r*")

# -------------------------------------------------定义图结构------------------------------------------------------------
graph = tf.Graph()
with graph.as_default():
    # ------------------------------------构造单层LSTM------------------------------------------
    # 声明输入数据
    X_p = tf.placeholder(dtype=tf.float32, shape=(None, TIME_STEPS, 1), name="input_placeholder")
    y_p = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="pred_placeholder")
    # lstm模块
    lstm_cell1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_1)
    lstm_cell2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_2)
    lstm_cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2])

    # 全零初始化初始状态
    init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)

    # # 手动初始化state
    # # 第一层state
    # lstm_layer1_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS_1))
    # lstm_layer1_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS_1))
    # layer1_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer1_c, h=lstm_layer1_h)
    # # 第二层state
    # lstm_layer2_c = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS_2))
    # lstm_layer2_h = tf.zeros(shape=(BATCH_SIZE, HIDDEN_UNITS_2))
    # layer2_state = tf.contrib.rnn.LSTMStateTuple(c=lstm_layer2_c, h=lstm_layer2_h)
    #
    # init_state = (layer1_state, layer2_state)

    # LSTM展开
    # dynamic rnn
    outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_p, initial_state=init_state, dtype=tf.float32)
    # 取最后一个时间步的输出作为最后的模型输出结果
    h = outputs[:, -1, :]

    # # 手动按照max_time展开
    # outputs = []
    # state = init_state
    # with tf.variable_scope('RNN'):
    #     for timestep in range(TIME_STEPS):
    #         if timestep > 0:
    #             tf.get_variable_scope().reuse_variables()
    #         # 每一个cell产生的state都作为下一个cell的输入cell
    #         (celloutput, state) = lstm_cell(X_p[:, timestep, :], state)
    #         outputs.append(celloutput)
    # print(len(outputs))
    # h = outputs[-1]
    # print(h.shape)

    # ------------------------------------定义loss和optimizer------------------------------------------
    mse = tf.losses.mean_squared_error(labels=y_p, predictions=h)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss=mse)
    init = tf.global_variables_initializer()

# -------------------------------------------------定义Session------------------------------------------------------------
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    for epoch in range(1, EPOCH + 1):
        results = np.zeros(shape=(TEST_EXAMPLES, 1))
        train_losses = []
        test_losses = []
        for j in range(TRAIN_EXAMPLES // BATCH_SIZE):
            _, train_loss = sess.run(
                [optimizer, mse],
                feed_dict={
                    X_p: X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                    y_p: y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                }
            )
            train_losses.append(train_loss)
        if epoch % 10 == 0:
            print("epoch:%d ,average training loss=%.4f" % (epoch, sum(train_losses) / len(train_losses)))

        for j in range(TEST_EXAMPLES // BATCH_SIZE):
            result, test_loss = sess.run(
                fetches=(h, mse),
                feed_dict={
                    X_p: X_test[j * BATCH_SIZE:(j + 1) * BATCH_SIZE],
                    y_p: y_test[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
                }
            )
            results[j * BATCH_SIZE:(j + 1) * BATCH_SIZE] = result
            test_losses.append(test_loss)
        if epoch % 10 == 0:
            print("epoch:%d ,average test loss=%.4f" % (epoch, sum(test_losses) / len(test_losses)))
            plt.plot(range(1000), results[:1000, 0])
    plt.show()

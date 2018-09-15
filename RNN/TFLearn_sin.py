import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

learn = tf.contrib.learn

# 1. 设置神经网络的参数
HIDDEN_SIZE = 30
NUM_LAYERS = 2

TIMESTEPS = 10
TRAINING_STEPS = 3000
BATCH_SIZE = 32

TRAINING_EXAMPLES = 10000
TESTING_EXAMPLES = 1000
SAMPLE_GAP = 0.01


# 2. 定义生成正弦数据的函数
def generate_data(seq):
    X = []
    y = []
    # 获取前TIMESTEPS个点的信息作为X，预测第i+TIMESTEPS这个时间的点的函数值作为y
    for i in range(len(seq) - TIMESTEPS - 1):
        X.append([seq[i: i + TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
'''
数据生成模拟
arr = np.arange(20)
arr_x, arr_y = generate_data(arr)
print(arr_x)
print(arr_y)
[[[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]]

 [[ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]]

 [[ 2.  3.  4.  5.  6.  7.  8.  9. 10. 11.]]

 [[ 3.  4.  5.  6.  7.  8.  9. 10. 11. 12.]]

 [[ 4.  5.  6.  7.  8.  9. 10. 11. 12. 13.]]

 [[ 5.  6.  7.  8.  9. 10. 11. 12. 13. 14.]]

 [[ 6.  7.  8.  9. 10. 11. 12. 13. 14. 15.]]

 [[ 7.  8.  9. 10. 11. 12. 13. 14. 15. 16.]]

 [[ 8.  9. 10. 11. 12. 13. 14. 15. 16. 17.]]]
[[10.]
 [11.]
 [12.]
 [13.]
 [14.]
 [15.]
 [16.]
 [17.]
 [18.]]
'''


# 定义Lstm单元
def LstmCell():
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE, state_is_tuple=True)

    return lstm_cell


# 3. 定义lstm模型
def lstm_model(X, y):
    cell = tf.contrib.rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])

    output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    # 得到max_time中的最后的结果，作为预测输出
    output = output[:, -1, :]
    # 因为输出的结果格式为(batch_size, hidden_size)，所以想要让结果输出为(batch_size, 1) 可以使用全连接层来实现
    predictions = tf.layers.dense(inputs=output, units=1)
    # 将predictions和labels调整统一的shape
    labels = tf.reshape(y, [-1])
    predictions = tf.reshape(predictions, [-1])
    # mse误差
    loss = tf.losses.mean_squared_error(predictions, labels)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adagrad',
        learning_rate=0.1
    )

    return predictions, loss, train_op


if __name__ == '__main__':
    # 自定义封装LSTM
    regressor = learn.Estimator(model_fn=lstm_model)

    # 生成数据
    test_start = TRAINING_EXAMPLES * SAMPLE_GAP
    test_end = (TRAINING_EXAMPLES + TESTING_EXAMPLES) * SAMPLE_GAP
    train_X, train_y = generate_data(np.sin(np.linspace(0, test_start, TRAINING_EXAMPLES, dtype=np.float32)))
    test_X, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES, dtype=np.float32)))
    train_X = np.reshape(train_X, newshape=(-1, TIMESTEPS, 1))
    test_X = np.reshape(test_X, newshape=(-1, TIMESTEPS, 1))

    # 调用fit函数训练模型
    regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)

    # 预测结果
    prediction = regressor.predict(test_X)
    # 因为生成的数据是generator，所以需要将其转化为list格式
    prediction = [x for x in prediction]
    # 计算mse
    mse = mean_squared_error(y_true=test_y, y_pred=prediction)
    print('Mean Square Error is: {}'.format(mse))

    # 对预测的sin函数曲线进行绘图
    plot_predicted, = plt.plot(prediction, label='predicted', color='red')
    plot_test, = plt.plot(test_y, label='real_sin', color='green')
    plt.legend([plot_predicted, plot_test], ['predicted', 'real_sin'])
    plt.show()

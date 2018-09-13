import numpy as np

# 样本每个时刻是输入
X = [1, 2]
# 初始化隐藏层的输入
state = [0.0, 0.0]

# 定义循环体中的隐藏层输入对应的权重矩阵
w_cell_state = np.asarray([[0.1, 0.2], [0.3, 0.4]])
# 定义循环体中的样本输入对应的权重矩阵
w_cell_input = np.asarray([0.5, 0.6])
# 定义循环体中的偏置
b_cell = np.asarray([0.1, -0.1])

# 定义输出层的权重矩阵
w_output = np.asarray([[1.0], [2.0]])
b_output = 0.1

# 按照时间顺序来执行前向传播
for i in range(len(X)):
    # 计算循环体中的全链接神经网络
    before_activation = np.dot(state, w_cell_state) + X[i] * w_cell_input + b_cell
    state = np.tanh(before_activation)

    # 当前时刻最终的输出
    final_output = np.dot(state, w_output) + b_output

    # 打印结果
    print(i+1)
    print('before activation:', before_activation)
    print('state:', state)
    print('output:', final_output)
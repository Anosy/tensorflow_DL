import tensorflow as tf
import numpy as np

# 单层的LSTM
# inputs = tf.placeholder(np.float32, shape=(32,40,5)) # 32 是 batch_size, 40为最大的时间长度，5为embedding_size 词嵌入长度
# lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
#
# print("output_size:",lstm_cell_1.output_size)
# print("state_size:",lstm_cell_1.state_size)
# output,state=tf.nn.dynamic_rnn(
#     cell=lstm_cell_1,
#     inputs=inputs,
#     dtype=tf.float32
# )
#
# print("output.shape:",output.shape)
# print("len of state tuple",len(state))
# print("state.h.shape:",state.h.shape)
# print("state.c.shape:",state.c.shape)


# # 多层的LSTM
inputs = tf.placeholder(np.float32, shape=(32,40,5)) # 32 是 batch_size
lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=128)
lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(num_units=256)
lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(num_units=512)
lstm_cell=tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])

print("output_size:",lstm_cell.output_size)
print("state_size:",lstm_cell.state_size)
output,state=tf.nn.dynamic_rnn(
    cell=lstm_cell,
    inputs=inputs,
    dtype=tf.float32
)

print("output.shape:",output.shape)

# 单层双向LSTM
# inputs = tf.placeholder(np.float32, shape=(32,40,5))
# lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=128)
# lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=120)
# print("output_fw_size:",lstm_cell_fw.output_size)
# print("state_fw_size:",lstm_cell_fw.state_size)
# print("output_bw_size:",lstm_cell_bw.output_size)
# print("state_bw_size:",lstm_cell_bw.state_size)
#
# output,state=tf.nn.bidirectional_dynamic_rnn(
#     cell_fw=lstm_cell_fw,
#     cell_bw=lstm_cell_bw,
#     inputs=inputs,
#     dtype=tf.float32
# )
# output_fw = output[0]
# output_bw = output[1]
# state_fw = state[0]
# state_bw = state[1]
#
# print("output_fw.shape:",output_fw.shape)
# print("output_bw.shape:",output_bw.shape)
# print("len of state tuple",len(state_fw))
# print("state_fw:",state_fw)
# print("state_bw:",state_bw)
#
# state_h_concat=tf.concat(values=[state_fw.h,state_bw.h],axis=1)
# print("state_fw_h_concat.shape",state_h_concat.shape)
#
# state_c_concat=tf.concat(values=[state_fw.c,state_bw.c],axis=1)
# print("state_fw_h_concat.shape",state_c_concat.shape)
#
# state_concat=tf.contrib.rnn.LSTMStateTuple(c=state_c_concat,h=state_h_concat)
# print(state_concat)
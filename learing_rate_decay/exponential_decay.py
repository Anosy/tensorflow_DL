# 指数衰减学习率，使模型在后期的学习更加的稳定
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


learning_rate = 0.1  # 初始学习速率时0.1
decay_rate = 0.96  # 衰减率
global_steps = 1000  # 总的迭代次数
decay_steps = 100  # 衰减次数

global_ = tf.Variable(tf.constant(0))
c = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=True)
d = tf.train.exponential_decay(learning_rate, global_, decay_steps, decay_rate, staircase=False)

T_C = []
F_D = []

with tf.Session() as sess:
    for i in range(global_steps):
        T_c = sess.run(c, feed_dict={global_: i})
        T_C.append(T_c)
        F_d = sess.run(d, feed_dict={global_: i})
        F_D.append(F_d)
plt.figure()
plt.plot(range(global_steps), F_D, 'r-', label='non-staircase')# "-"表示折线图,r表示红色,b表示蓝色
plt.plot(range(global_steps), T_C, 'b-', label='staircase')
plt.legend()
plt.show()

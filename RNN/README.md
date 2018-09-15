# 本部分主要介绍的利用tensorflow来实现RNN、LSTM

## 第一部分，RNN前向传播的实现，具体代码见：simple_rnn_forward.py
**实现的效果见下图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/RNN/my_picture/rnn_forward.jpg)<br>

## 第二部分，LSTM基本结构，以及使用tensorflow来搭建LSTM。具体的代码见：LSTM,.py
**LSTM的基本结构如下图所示：** <br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/RNN/my_picture/LSTM3.png)<br>
### 单层LSTM
核心代码：<br>
1.  inputs = tf.placeholder(np.float32, shape=(32,40,5))  32表示batch_size，40表示max_time也就是最大的时间序列长度，5表示embedding_size<br>
2.  lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(num_units=128) 定义LSTM的基本单元，其中隐藏单元的维度为128<br>
3.
        output,state=tf.nn.dynamic_rnn( 此函数会通过max_time，将网络按照时间展开
        cell=lstm_cell_1,
        inputs=inputs,
        dtype=tf.float32
        )
        
### 多层LSTM
核心代码：<br>
lstm_cell=tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1,lstm_cell_2,lstm_cell_3])将多个cell给向上来堆叠
### 单层双向LSTM
1.
        lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(num_units=128)定义了正向的cell
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(num_units=128)定义了反向的cell
2.
        output,state=tf.nn.bidirectional_dynamic_rnn(将正向的cell和反向的cell合并形成一个网络，并且将其按照时间来展开
            cell_fw=lstm_cell_fw,
            cell_bw=lstm_cell_bw,
            inputs=inputs,
            dtype=tf.float32
        )

## 第三部分，使用单层LSTM来预测sin函数，具体代码见：predict_sin.py
核心代码：<br>

        # ------网络结构-------
        lstm_cell = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS)  # 声明LSTM单元
        init_state = lstm_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)  # 零初始化状态
        outputs, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=X_p, initial_state=init_state, dtype=tf.float32)  # 将LSTM按照max_time时间给展开
        h = outputs[:, -1, :]  # 最后一个时间步的输出作为该LSTM层网络的输出

150次迭代结果：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/RNN/result_picture/onelayer_lstm.png)<br>

## 第四部分，使用多层LSTM来预测sin函数，具体代码见：predict_sin_MultiRnnCell.py
核心代码：<br>

        lstm_cell1 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_1)
        lstm_cell2 = rnn.BasicLSTMCell(num_units=HIDDEN_UNITS_2)
        lstm_cell = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell1, lstm_cell2])

50次迭代的结果：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/RNN/result_picture/multi_lstm.png)<br>
**可以发现，多层LSTM即使只有50次迭代，其效果就很逼近sin函数**
### 添加手动初始化和手动展开LSTM
核心代码：<br>

        outputs = list()                                   
        state = init_state
        with tf.variable_scope('RNN'):
        for timestep in range(TIME_STEPS):
            if timestep > 0:
                tf.get_variable_scope().reuse_variables()
            # 每一个cell产生的state都作为下一个cell的输入cell
            (cell_output, state) = multi_lstm(X_p[:, timestep, :], state)
            outputs.append(cell_output)
        h = outputs[-1]
        
## 第六部分，使用LSTM来实现MNIST分类，具体代码见：LSTM_for_Mnist.py
核心代码：<br>
1. rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=NUM_UNITS)  定义lstm cell<br>
2.
        outputs, final_state = tf.nn.dynamic_rnn(   将lstm按照max_time给展开
            cell=rnn_cell,
            inputs=image,
            initial_state=None,
            dtype=tf.float32,
            time_major=False
        )
3. output = tf.layers.dense(inputs=outputs[:, -1, :], units=N_CLASSES)  全连接层，将输出的结果给转化为(batch_size, N_CLASSES)<br>
4. loss = tf.losses.softmax_cross_entropy(onehot_labels=train_y, logits=output)   定义损失函数<br>
5. train_op = tf.train.AdamOptimizer(LR).minimize(loss)   定义优化方法<br>
6.
        correct_prediction = tf.equal(tf.argmax(train_y, axis=1), tf.argmax(output, axis=1))  正确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
最终结果<br>
train loss: 2.3051 | test accuracy: 0.13<br>
train loss: 0.0936 | test accuracy: 0.97<br>
train loss: 0.0224 | test accuracy: 0.98<br>
train loss: 0.0536 | test accuracy: 0.98<br>
train loss: 0.0758 | test accuracy: 0.97<br>
train loss: 0.0862 | test accuracy: 0.98<br>
train loss: 0.0193 | test accuracy: 0.98<br>
train loss: 0.0078 | test accuracy: 0.98<br>
train loss: 0.0549 | test accuracy: 0.99<br>
train loss: 0.1516 | test accuracy: 0.99<br>
train loss: 0.1834 | test accuracy: 0.98<br>
train loss: 0.0998 | test accuracy: 0.98<br>
train loss: 0.0051 | test accuracy: 0.98<br>
train loss: 0.0455 | test accuracy: 0.99<br>
train loss: 0.0566 | test accuracy: 0.98<br>
train loss: 0.1219 | test accuracy: 0.98<br>
虽然强行使用lstm来搭建MNIST分类系统，但是效果还行！<br>

## 第七部分，使用LSTM的TFLearn接口来实现对sin函数的预测
核心代码：<br>

        1.cell = tf.contrib.rnn.MultiRNNCell([LstmCell() for _ in range(NUM_LAYERS)])  # 声明多层的LSTM神经网络
        2.output, _ = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)   # 按照时间来展开
        3.output = output[:, -1, :]  # 将得到的最后的输出作为最终的输出
        4.predictions = tf.layers.dense(inputs=output, units=1)  # 将LSTM网络输出的结果给导入到全连接网络，得到最后的输出结果为shape=()
        5.loss = tf.losses.mean_squared_error(predictions, labels)  # 定义损失函数mse
        6.创建优化器并且得到优化步骤
        train_op = tf.contrib.layers.optimize_loss(
            loss,
            tf.contrib.framework.get_global_step(),
            optimizer='Adagrad',
            learning_rate=0.1
            )
        7.regressor = learn.Estimator(model_fn=lstm_model)  # 自定义封装lstm
        8.regressor.fit(train_X, train_y, batch_size=BATCH_SIZE, steps=TRAINING_STEPS)  # 调用fit函数来训练模型
        9.prediction = regressor.predict(test_X) # 预测结果
得到的结果如下：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/RNN/result_picture/TFLearn_for_sin.png)



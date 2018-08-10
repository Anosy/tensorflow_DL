'''
完整的mnist程序的流程：
1. 加载数据
2. 设置模型的参数
3. 定义前向传播函数(分为滑动平均和非滑动平均)
4. 训练过程
   1) 定义输入和输出
   2) 生成隐藏层的参数(初始化)
   3) 利用前向传播函数来计算输出的y(不包含滑动平均)，训练时的输出
   4) 初始化滑动平均类，并且利用前向传播函数来输出average_y， 验证/测试时候的输出
   5) 计算损失函数，这里使用的是sparse_softmax_cross_entropy_with_logits，其对与单分类问题的来说计算的速度快了，
      且这里的使用的label为非稀疏的，在这里使用的计算的过程中，得到的结果的shape=(batch_size, 1)，所以计算的时候要mean操作
   6) 考虑正则化，将之前计算的交叉熵损失函数加上正则项，从而可以得到总的损失
   7) 学习率指数衰减
   8) 定义模型的训练过程，其中利用control_dependencies，来将训练步骤和参数的滑动平均合为一步进行执行
   9) 计算模型的精确度。在计算的过程中，由于y_和y都是使用了one-hot编码之后的数，所以在判断是哪一类的时候需要执行argmax操作，
      同时也需要进行mean操作，目的是取整个batch的平均的正确率
   10) 定义了两种的训练模型的方法。具体为：开启会话/获取数据/迭代循环/sess.run()来启动训练，这里每次循环都使用有个batch
    的大小来进行训练。第二种方法的区别在于，其每次迭代的过程种都使用了batch，但是当全部的数据都过一遍后，才进行下一次循环迭代
    
    注意：在考虑滑动平均参数的时候，一定要考虑到在训练的时候使用的是没有滑动的参数来训练迭代，但是在预测的时候使用的是滑动平均处理后的结果
      
   
'''
import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
xs, ys = mnist.train.next_batch(100)
import tensorflow as tf
import matplotlib.pyplot as plt

# MNIST数据集的参数
INPUT_NODE = 784  # 输入的节点数
OUTPUT_NODE = 10  # 输出的节点数

# 配置神经网络的参数
LAYER1_NODE = 500  # 神经网络的隐藏层的个数
BATCH_SIZE = 100  # 训练一个batch中的训练数据个数
LEARNING_RATE_BASE = 0.8  # 基础的学习率
LEARNING_RATE_DECAY = 0.99  # 模型正则化前面的系数
REGULARIZATION = 0.0001  # 描述模型复杂的正则化损失函数中的系数
TRAINING_STEPS = 30000  # 训练的轮数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均的衰减

# 定义前向传播函数
def inference(input_tensor, avg_class, weight1, biases1, weight2, biases2):
    # 当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
        layer2 = tf.matmul(layer1, weight2) + biases2

        return layer2
    # 当提供滑动平均类的时候，先用函数计算出变量的滑动平均值，然后再进行滑动平均
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1))
        layer2 = tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)

        return layer2


# 训练模型的过程
def train(mnist):
    # 定义输入和输出
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层的参数
    weight1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    # 生成输出层的参数
    weight2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))
    # 计算当前参数下神经网络的计算结果，这里的y指的是没有进行滑动平均的结果
    y = inference(x, None, weight1, biases1, weight2, biases2)
    # 定义变量储存训练的轮数的变量，但是这个变量是不需要滑动平均的，所以也就不需要进行训练
    global_step = tf.Variable(0, trainable=False)
    # 初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 对可以训练的变量，进行滑动平均
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 计算滑动平均后的前向传播结果
    average_y = inference(x, variable_averages, weight1, biases1, weight2, biases2)
    # 定义损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))  # label要为非稀疏的，因为其默认就对其进行稀疏操作
    # 由于这里使用了batch，所以在计算的结果后要计算batch的平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算l2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION)
    regularization = regularizer(weight1) + regularizer(weight2)
    # 总的损失loss
    loss = cross_entropy_mean + regularization
    # 设置学习率衰减
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
                                               LEARNING_RATE_DECAY)
    # 训练步骤
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    # 在训练模型过程中，每过一遍数据都要进行反向传播和更新参数, 而且还要进行滑动平均，所以需要一次完成
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')
    # 计算精确度
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 第一种训练的方式，该过程强制给的训练模型训练次数30000
    test_acc_dict = {}
    validate_acc_dict = {}
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = accuracy.eval(feed_dict=validate_feed)
                print('After %d training steps, validation accurary is %g' % (i, validate_acc))
                # print(sess.run(global_step))  # 打印当前的迭代次数
                # 绘制测试集和验证集的准确的结果
                test_acc = accuracy.eval(feed_dict=test_feed)
                test_acc_dict[i] = test_acc
                validate_acc = accuracy.eval(feed_dict=validate_feed)
                validate_acc_dict[i] = validate_acc
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})
        plt.plot(test_acc_dict.keys(), test_acc_dict.values(), 'r', label='test')
        plt.plot(validate_acc_dict.keys(), validate_acc_dict.values(), 'g', label='validate')
        plt.legend()
        plt.grid(ls='-.')
        # print('After %d test steps, test accuracy is %g' % (TRAINING_STEPS, test_acc))
        plt.show()

    # 第二种方式来训练模型，该方式训练的次数为total_batch*60次的训练次数
    # with tf.Session() as sess:
    #     tf.global_variables_initializer().run()
    #     for epoch in range(60):
    #         average_loss = 0.
    #         total_batch = int(mnist.train.num_examples / BATCH_SIZE)
    #         for i in range(total_batch):
    #             batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
    #             _, l = sess.run([train_op, loss], feed_dict={x: batch_x, y_: batch_y})
    #             average_loss += l / total_batch
    #         if epoch % 10 == 0:
    #             print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(average_loss))
    #     test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    #     test_acc = accuracy.eval(feed_dict=test_feed)
    #     print('After %d train steps, test accuracy is %g' % (sess.run(global_step), test_acc))




if __name__ == '__main__':

    train(mnist)

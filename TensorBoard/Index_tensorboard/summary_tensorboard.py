import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = "./log"
BATCH_SIZE = 100
TRAIN_STEPS = 30000


# var给出了需要记录的张量,name给出了在可视化结果中显示的图表名称，这个名称一般和变量名一致
def variable_summaries(var, name):
    # 将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope('summaries'):
        # 通过tf.histogram_summary函数记录张量中元素的取值分布
        tf.summary.histogram(name, var)

        # 计算变量的平均值，并定义生成平均值信息日志的操作
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        # 计算变量的标准差，并定义生成其日志文件的操作
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)

# 生成一层全链接的神经网络。
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在一个统一的命名空间下
    with tf.name_scope(layer_name):
        # 声明神经网络边上的权值，并调用权重监控信息日志的函数
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')

        # 声明神经网络边上的偏置，并调用偏置监控信息日志的函数
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络节点输出在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations

def main():
    mnist = input_data.read_data_sets('./Mnist_data', one_hot=True)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 将输入给还原成图片矩阵，并且通过tf.summary.image函数写入图片信息
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)  # 最多输出多少个图像

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)  # tf.identity表示返回一个一模一样的tensor的op，会在graph上添加一个节点

    # 计算交叉熵, 并且生成日志
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))  # reduce_mean 计算batch 中样本的平均损失
        tf.summary.scalar('cross entropy', cross_entropy)


    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 计算当前节点的正确率，并且生成日志
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
    # 写入日志的函数如tf.summary.scalar和tensorflow中都一样，都需要通过sess.run才能运行，但是一一调用太麻烦，所以tf.merge_all()函数来将所有
    # 之前定义的写入日志操作都给统一起来
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 初始化写日志的writer
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph())

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 生成日志以及训练过程
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_:ys})
            # 将所有生成的日志给写入到文件
            summary_writer.add_summary(summary, global_step=i)
    summary_writer.close()

if __name__ == '__main__':
    main()










import tensorflow as tf
import numpy as np

files = tf.train.match_filenames_once("/path/to/data.tfrecords-*")  # 此函数获取一个符合正则表达式的所有文件
file_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(file_queue)
# 解析数据
features = tf.parse_single_example(serialized_example, features={
    'image': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.int64),
    'height': tf.FixedLenFeature([], tf.int64),
    'width': tf.FixedLenFeature([], tf.int64),
    'channel': tf.FixedLenFeature([], tf.int64),
})
image, label = features['image'], features['label']
height, width = features['heiht'], features['width']
channel = features['channel']
image_decode = tf.decode_raw(image, tf.uint8)
image_decode.set_shape([height, width, channel])

# 假设神经网络输入层的图片大小为300
image_size = 300
distorted_image = preprocess_for_train(image_decode, image_size, image_size, None)

# 将处理过后的图像和标签数据通过tf.train.shuffle_batch 整理成神经网络训练时需要的batch
min_after_dequeue = 1000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
# tf.train.shuffle_batch函数的入队操作就是数据处理以及预处理的过程
image_batch, label_batch = tf.train.shuffle_batch([distorted_image, label], batch_size, capacity, min_after_dequeue,
                                                  num_threads=5)

# 定义神经网络的优化结构以及优化过程
logit = inference(image_batch)
loss = cal_loss(loss, label_batch)

train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord=coord)
    for i in range(TRAINING_EPOCHS):
        _, loss = sess.run([train_op, loss])

    # -------停止线程
    coord.request_stop()
    coord.join(threads)

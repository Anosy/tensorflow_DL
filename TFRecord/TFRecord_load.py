import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的样例
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['./TFRecord_Output/output.tfrecords'])
# 从文件中读出一个样例
_, serialized_example = reader.read(filename_queue)
# 解析读入的一个样例
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw 可以将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)
# 启动多线程来处理输入的数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for _ in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(image.shape, label, pixel)

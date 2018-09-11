import tensorflow as tf
import numpy as np

def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)

    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    # # 查看是否存在标注框,如果没有表示整个图片就是需要关注的部分
    # if bbox is None:
    #     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # # 转化图片张量的类型
    # if image.dtype != tf.float32:
    #     image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # # 随机的截取图片中一个块。
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
    #     tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # 将随机截取的图片调整为神经网络的输入层的大小
    distorted_image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))
    # 随机左右翻转图片
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    # 使用一种颜色调整函数
    distorted_image = distort_color(distorted_image, np.random.randint(1))

    return distorted_image

if __name__ == '__main__':

    # 创建文件列表，并且通过列表创建输入文件队列。这里的文件夹目录为Mnist_Output
    files = tf.train.match_filenames_once('./Mnist_Output/output.tfrecords')
    filename_queue = tf.train.string_input_producer(files, shuffle=False)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

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
    print(sess.run(images).shape)
    # # 定义神经网络输入层图片的大小
    # image_size = 299
    # # 使用图像处理函数来处理输入的图像
    # distorted_image = preprocess_for_train(images, image_size, images, None)
    # # 使用tf.train.shuffle_batch来生成batch
    # min_after_dequeue = 10000
    # batch_size = 100
    # capacity = min_after_dequeue + 3 * batch_size
    # image_batch, label_batch = tf.train.shuffle_batch([distorted_image, labels], batch_size=batch_size, capacity=capacity,
    #                                                   min_after_dequeue=min_after_dequeue)

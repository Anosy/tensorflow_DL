import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

# 生成整数型的属性
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# 生成字符串的属性
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    mnist = input_data.read_data_sets('./MNIST_data', dtype=tf.uint8, one_hot=True)
    # 获取数据集信息：图片，标签，像素，数量
    images = mnist.train.images
    labels = mnist.train.labels
    pixels = images.shape[1]
    num_examples = mnist.train.num_examples

    # 输出TFRecord文件的地址
    filename = "./TFRecord_Output/output.tfrecords"
    if not os.path.exists('./TFRecord_Output/'):
        os.makedirs('./TFRecord_Output/')
    # 创建一个writer来写TFRecord文件
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        # 将图片矩阵给转化为一个字符串
        image_raw = images[index].tostring()
        # 将一个样例转化为Example Protocol Buffer ,并且将所有的信息给写入到数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'image_raw': _bytes_feature(image_raw)
        }))
        # 将一个Example写入到TFRecord文件中
        writer.write(example.SerializeToString())
    writer.close()

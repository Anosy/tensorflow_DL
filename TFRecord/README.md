# 本部分主要介绍的是tensorflow中的TFRecord

## TFRecord数据文件
tfrecord数据文件是一种将图像数据和标签统一存储的二进制文件，能更好的利用内存，在tensorflow中快速的复制，移动，读取，存储等。<br>

## 第一部分，将数据给保存成TFRecord格式，代码见:TFRecord_save.py
主要内容：<br>
1. 创建一个writer来写TFRecord文件<br>

        writer = tf.python_io.TFRecordWriter(file_name)

2. 将所有的信息给转化为TFRcord数据结构

        example = tf.train.Example(features=tf.train.Features(feature={
                    'pixels': tf.train.Feature(int64_list=tf.train.Int64List(value=[pixels])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels])),
                    'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                }))

3. 将信息给写入到文件中

        writer.write(example.SerializeToString())

## 第二部分，读取TFRcord格式文件，代码见: TFRecord_load.py 
主要内容：<br>
1. 创建一个reader来读取TFRecord文件中的样例<br> 

        reader = tf.TFRecordReader()

2. 创建一个队列来维护输入文件列表<br>

        filename_queue = tf.train.string_input_producer(['./TFRecord_Output/output.tfrecords'])

3. 从文件中读出一个样例<br>

        _, serialized_example = reader.read(filename_queue)

4. 解析读入的一个样例

        features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64)
        })

5. 转换格式
6. 开启session
7. 启动多线程来处理输入数据


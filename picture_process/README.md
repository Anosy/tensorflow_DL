# 本部分主要介绍的tensorflow中的图像预处理部分
## 1. process.py 介绍的是图像处理的各个函数，以及其的作用
## 2. full_process.py介绍的是综合全部的图像处理方法的一个完整的样例
### 主要函数介绍：
1.image = tf.image.random_brightness(image, max_delta=32. / 255.)   调整图片的亮度<br>
2.image = tf.image.random_saturation(image, lower=0.5, upper=1.5)   调整图片的饱和度<br>
3.image = tf.image.random_hue(image, max_delta=0.5)                 调整图片的色相<br>
4.image = tf.image.random_contrast(image, lower=0.5, upper=1.5)     调整图片的对比度<br>
5.image = tf.image.convert_image_dtype(image, dtype=tf.float32)     调整函数数据类型<br>
6.resized = tf.image.resize_images(img_data, [300, 300], method=0)  调整图片的大小<br>
7.ud_flipped = tf.image.flip_up_down(img_data)                      调整图片的上下颠倒<br>
8.img_data = tf.image.decode_jpeg(image_raw_data)                   将图片给解码成三维矩阵<br>
9.begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)                   随机选择标注框的绝对位置<br>
10.distorted_image = tf.slice(img_data, begin, size)                取出标注框的图片内容<br>
最终得到的结果为：
![](https://github.com/Anosy/tensorflow_DL/blob/master/inception-v3_transfer-learning/picture/Results_from_six_situations.png)<br>

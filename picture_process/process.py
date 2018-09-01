import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# 读取原始图像数据
image_raw_data = tf.gfile.FastGFile('./picture/lena.jpg', 'rb').read()

with tf.Session() as sess:
    # 将图片给解码，使用tf.image.decode_jpeg函数来对jpg格式的图片进行解码
    img_data = tf.image.decode_jpeg(image_raw_data)
    # print('raw_data_size:\n', img_data.eval().shape)
    # print('raw_data:\n', img_data.eval())

    # 使用pyplot工具来可视化
    # plt.imshow(ima_data.eval())

    # 将数据的类型给转化为实数型，方便进行图像处理
    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    # 将三维矩阵按照jpeg格式来重新编码，保存成文件
    # encode_image = tf.image.encode_jpeg(ima_data)
    # with tf.gfile.GFile('./picture/lena_encode.jpg', 'wb') as f:
    #     f.write(encode_image.eval())

    # 图片大小调整
    # 通过tf.image.resize_images来调整图片的大小,函数的第一个参数为原始图片，第二个参数为调整后的图片的大小，第三个为调整图片的方法
    # method=0,1,2,3  分别对应了双线性插值法，最近邻法，双三次插值法，面积插值法
    resized = tf.image.resize_images(img_data, [300, 300], method=0)
    # 由于处理后的函数的格式为float32格式的，所以需要转化为uint8才能打印图片
    resized = np.asarray(resized.eval(), dtype='uint8')
    # plt.imshow(resized)
    # plt.show()
    # 通过tf.image.resize_image_with_crop_or_pad 函数来调整图像的大小，主要是通过剪切和填充
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 300, 300)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    # plt.imshow(croped.eval())
    # plt.imshow(padded.eval())
    # plt.show()

    # 图片翻转
    # 将图片给上下翻转
    ud_flipped = tf.image.flip_up_down(img_data)
    # 将图片给左右翻转
    lr_flipped = tf.image.flip_left_right(img_data)
    # 将图片按对角线翻转
    transposed = tf.image.transpose_image(img_data)
    # 按一定概率上下，左右旋转
    ud_random_flipped = tf.image.random_flip_up_down(img_data)
    lr_random_flipped = tf.image.random_flip_left_right(img_data)
    # plt.imshow(ud_random_flipped.eval())
    # plt.show()

    # 图片色彩调整
    # 图片的亮度调整
    adjusted_brightness = tf.image.adjust_brightness(img_data, 0.5)
    # 在[-max_delta, max_delta]范围内随机改变图片的亮度
    max_delta = 0.5
    adjusted_brightness_range = tf.image.random_brightness(img_data, max_delta)
    # 调整图片的对比度
    adjusted_contrast = tf.image.adjust_contrast(img_data, -5)
    # 调整图片的色相
    adjusted_hue = tf.image.adjust_hue(img_data, 0.6)
    # 调整图片的饱和度
    adjusted_saturation = tf.image.adjust_saturation(img_data, 5)
    # 图像标准化，白化过程
    adjusted_whitening = tf.image.per_image_standardization(img_data)
    # plt.imshow(adjusted_whitening.eval())
    # plt.show()

    # 处理标注框
    # 标注框的相对位置
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    # 随机选择标注框的绝对位置
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)
    # 因为tf.image.convert_image_dtype处理的图片都是四维的，需要加上一个维度
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    # 绘制标注框
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)
    distorted_image = tf.slice(img_data, begin, size)
    # plt.imshow(image_with_box[0].eval())  # 绘制出具有标注框的图片
    # plt.imshow(distorted_image.eval())
    # plt.show()


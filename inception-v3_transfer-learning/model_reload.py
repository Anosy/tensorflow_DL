import numpy as np
import tensorflow as tf
from transfer_inception_v3 import creat_image_lists

image_path = ['./my_picture/rose.jpg',
              './my_picture/sunflower.jpg']
# 获取目录中最新模型的文件名
ckpt = tf.train.get_checkpoint_state('./model/')
# .meta文件保存当前的图结构，载入图模型
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
with tf.Session() as sess:
    # 加载参数
    saver.restore(sess, ckpt.model_checkpoint_path)
    g = tf.get_default_graph()
    for img in image_path:
        image_data = open(img, 'rb').read()
        # 通过Inception-v3的瓶颈层，输出图片对应的向量
        bottleneck = sess.run(g.get_tensor_by_name('import/pool_3/_reshape:0'),
                              feed_dict={g.get_tensor_by_name('import/DecodeJpeg/contents:0'): image_data})
        # 通过softmax输出结果
        class_result = sess.run(g.get_tensor_by_name('final_train_ops/Softmax:0'),
                                feed_dict={g.get_tensor_by_name('BottleneckInputPlaceholder:0'): bottleneck})
        images_lists = creat_image_lists(10, 10)
        print(images_lists.keys())
        print(np.squeeze(class_result))

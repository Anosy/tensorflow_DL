# 本部分主要介绍的Inception-v3模型，包括介绍其主要模块、利用训练好的Inception-v3模型进行迁移学习，以及学习好后的模型进行保存
## 文件结构如下：
![](https://github.com/Anosy/tensorflow_DL/blob/master/inception-v3_transfer-learning/picture/structure.png)<br>
其中模型下载的地址为：<br>
	http://download.tensorflow.org/example_images/flower_photos.tgz
	https://storage.googleapis.com/download.tensorflow.org/models/inception_dec_2015.zip

## 第一部分：Inception简介
在GoogleNet出现前，主流的神经网络存在三大问题：<br>
1.参数太多，容易过拟合，若训练数据集有限；<br>
2.网络越大计算复杂度越大，难以应用；<br>
3.网络越深，梯度越往后穿越容易消失（梯度弥散），难以优化模型<br>
Inception模块主要就是解决上面这些问题<br>
Inception模块的结构如下图所示：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/inception-v3_transfer-learning/picture/inception-structure.png)<br>

## 第二部分，Inception模块的搭建，具体的代码见  inception_v3.py
### 模块的结果如下
第一条路径：卷积层的深度为320，1x1的卷积核<br>
第二条路径：先通过卷积层深度为384，1x1的卷积，再通过1x3和3x1卷积的结合的卷积层，深度也为384。后者的目的是为了节省参数<br>
第三条路径：卷积层的深度为448，1x1 + 卷积层深度为384，3x3卷积 + 卷积层深度为384，1x3 3x1卷积<br>
第四条路径：平均池化，窗口大小为3x3 + 卷积层深度为192，1x1卷积<br>
最后：将每条路径给合并起来，使用net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])<br>

## 第三部分，迁移学习，代码见：transfer_inception_v3.py
### 代码结构：
1.creat_image_lists(testing_percentage, validation_percentage)  从所有数据文件夹中读取所有的图片，便且按训练、验证、测试数据分开<br>
2.get_image_path(image_lists, image_dir, label_name, index, category)  通过类别名称、所属数据集和图片编号获取一张图片的地址<br>
3.get_bottleneck_path(image_lists, label_name, index, category)  获取结果Inception_v3模型处理之后的特征向量文件地址<br>
4.run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor)  使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量<br>
5.get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor)  获取一张图片经过Inception-v3模型处理之后的特征向量，如果没有那么就先计算特征向量，然后保存到文件<br>
6.get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor)  随机获取一个batch作为训练数据<br>
7.get_test_bottlenecks(sess, image_lists, n_class, jpeg_data_tensor, bottleneck_tensor)  获取全部测试数据<br>
8.main()  主函数<br>
### 模型实现逻辑
先加载以及训练好的Inception-v3。加载的代码如下：<br>
	with gfile.FastGFile("./graph.pb",'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')
然后通过import_graph_def，返回瓶颈层的输出，以及模型的输入<br>
定义最后一层新的神经网络，全连接+softmax回归<br>
开启session。训练过程：先通过原始模型，计算出瓶颈层的输出，然后将输出在导入到新搭建的网络中。模型保存：每500次保存模型一次<br>
### 模型运行的结果：
	Step 0: Validation accuracy on random sampled 100 examples = 31.0%
	Step 100: Validation accuracy on random sampled 100 examples = 79.0%
	Step 200: Validation accuracy on random sampled 100 examples = 86.0%
	Step 300: Validation accuracy on random sampled 100 examples = 89.0%
	Step 400: Validation accuracy on random sampled 100 examples = 90.0%
	Step 500: Validation accuracy on random sampled 100 examples = 94.0%
	Step 600: Validation accuracy on random sampled 100 examples = 91.0%
	Step 700: Validation accuracy on random sampled 100 examples = 90.0%
	Step 800: Validation accuracy on random sampled 100 examples = 92.0%
	Step 900: Validation accuracy on random sampled 100 examples = 85.0%
	Step 1000: Validation accuracy on random sampled 100 examples = 95.0%
	Step 1100: Validation accuracy on random sampled 100 examples = 90.0%
	Step 1200: Validation accuracy on random sampled 100 examples = 91.0%
	Step 1300: Validation accuracy on random sampled 100 examples = 90.0%
	Step 1400: Validation accuracy on random sampled 100 examples = 84.0%
	Step 1500: Validation accuracy on random sampled 100 examples = 93.0%
	Step 1600: Validation accuracy on random sampled 100 examples = 89.0%
	Step 1700: Validation accuracy on random sampled 100 examples = 93.0%
	Step 1800: Validation accuracy on random sampled 100 examples = 90.0%
	Step 1900: Validation accuracy on random sampled 100 examples = 92.0%
	Step 1999: Validation accuracy on random sampled 100 examples = 92.0%
	Final test accuracy = 93.1%

## 第四部分，模型加载和测试，代码见：model.reload.py
### 主要部分：
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state('./model/')
        saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
        saver.restore(sess, ckpt.model_checkpoint_path)
注：由于该方法是也加载了图结构，所以加入g = tf.get_default_graph()，得到图结构，然后通过g.get_tensor_by_name来找到对应的tensor。
### 运行的结果如下：
	dict_keys(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
	[1.750127e-04 7.448361e-05 9.928653e-01 2.692707e-03 4.192517e-03]
	dict_keys(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'])
	[0.05063567 0.0603058  0.01655335 0.7938776  0.0786276 ]






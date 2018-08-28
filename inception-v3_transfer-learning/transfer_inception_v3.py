import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

BOTTLENECK_TENSOR_SIZE = 2048  # 瓶颈层节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # 瓶颈层输出张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 输入层张量名称

MODEL_DIR = './inception_dec_2015'  # 模型存放文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'  # 模型名

CACHE_DIR = './bottleneck1'  # 瓶颈输出中转文件夹，如果设置为bottleneck，二次运行会冲突
INPUT_DATA = './flower_photos'  # 数据文件夹

VALIDATION_PERCENTAGE = 10  # 验证用数据百分比
TEST_PERCENTAGE = 10  # 测试用数据百分比

# 神经网络参数设置
LEARNING_RATE = 0.01
STEP = 2000
BATCH = 100

# 这个函数从所有数据文件夹中读取所有的图片，便且按训练、验证、测试数据分开
def creat_image_lists(testing_percentage, validation_percentage):
    # 得到的所有图片都存在result字典中。字典的key为类别的名称，value也是一个字典，字典中存储了所有图片的图片名称
    result = {}
    # 获取当前目录下所有的子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前的目录,所以跳过
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下的所有有效的图片
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)  # 获取当前文件夹的名称
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))  # glob.glob模块用于查找指定的文件，便且将其以list的形式给返回
        if not file_list:
            continue

        # 通过目录的名称来获取类别
        label_name = dir_name.lower()
        # 初始化当前类别的训练集、测试数据和验证数据
        training_images = []
        testing_images = []
        validation_images = []

        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 返回path的最后的文件名。如：xxx.jpg
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据给放入到字典中
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    return result

# 通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists: 所有图片的信息  image_dir 参数给出了根目录  label_name 类别的名称
# index 给定了需要获取图片的编号  category 指定数据是属于训练集测试集还是验证集
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取指定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据所属的数据集的名称来获取全部的图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片的文件名
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    # 得到最终的地址
    full_path = os.path.join(image_dir, sub_dir, base_name)

    return full_path


# 获取结果Inception_v3模型处理之后的特征向量文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'


# 使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 卷积神经网络处理后的结果为一个四维的数组，需要将其给压缩成一维的数组
    bottleneck_values = np.squeeze(bottleneck_values)

    return bottleneck_values

# 获取一张图片经过Inception-v3模型处理之后的特征向量，如果没有那么就先计算特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片其对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    # 模型处理之后的特征向量文件地址  .\xxx\xxx\xxx.txt
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)

    # 如果这个文件的特征向量不存在，计算并且保存
    if not os.path.exists(bottleneck_path):
        # 获取原始图片的路径
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        # 获取图片的内容
        image_data = gfile.FastGFile(image_path, 'rb').read()
        # 通过Inception-v3 模型计算特征向量
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        # 将计算得到的向量给保存成文件，并且每个向量的元素用逗号分开
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件中获取图片相应的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

# 这个函数随机获取一个batch作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片编号，加入到训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        # 瓶颈层张量
        bottleneck = get_or_create_bottleneck(  # 获取对应标签随机图片瓶颈张量
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0  # 标准结果[0,0,1,0...]
        # 收集瓶颈张量和label
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

# 获取全部测试数据
def get_test_bottlenecks(sess, image_lists, n_class, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):  # 索引, {文件名}
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index,
                category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_class, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main():
    # 生成文件字典
    images_lists = creat_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    # 记录label种类(字典项数)
    n_classes = len(images_lists.keys())

    # 从pb文件加载模型
    '''模板：
    with gfile.FastGFile("./graph.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
    '''
    with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:   # 阅读器上下文
        graph_def = tf.GraphDef()  # 生成图
        graph_def.ParseFromString(f.read())  # 图加载模型
    # 加载图上节点张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(  # 从图上读取张量，同时导入默认图
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    '''新的神经网络'''
    # 输入层,由原模型输出层feed
    bottleneck_input = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32, [None, n_classes], name='GroundTruthInput')
    # 全连接层
    with tf.name_scope('final_train_ops'):
        Weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, Weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 当前的迭代次数
    global_step = tf.Variable(0, trainable=False)
    # 交叉熵损失函数
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth_input))
    # 优化算法选择
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step)

    # 正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # 模型保存目录
    if not os.path.exists("./model/"):
        os.makedirs("./model/")

    with tf.Session() as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEP):
            # 随机batch获取瓶颈输出 & label， Batch这里设置为100
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, images_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step,
                     feed_dict={bottleneck_input: train_bottlenecks, ground_truth_input: train_ground_truth})

            # 每迭代100次运行一次验证程序
            if i % 100 == 0 or i + 1 == STEP:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, images_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))
            # 每迭代500次保存一次训练的模型
            if i % 500 == 0 or i + 1 == STEP:
                saver.save(sess, "./model/TransferLearning.model", global_step=global_step)

        # 在最后的测试数据上测试准确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, images_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            bottleneck_input: test_bottlenecks, ground_truth_input: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))
if __name__ == '__main__':
    main()

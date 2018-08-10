import tensorflow as tf

# 定义模型的结构参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 定义获取模型权重的参数
def get_weight_variable(shape, regularizer):
    weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weight))  # 将正则化给加入到损失中
    return weight

# 前向传播
def inference(input_tensor, regularizer):
    with tf.variable_scope('layer1'):
        weight = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weight) + biases)

    with tf.variable_scope('layer2'):
        weight = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weight) + biases

    return layer2


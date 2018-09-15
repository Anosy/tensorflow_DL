import tensorflow as tf
from sklearn import model_selection
from sklearn import datasets
from sklearn import  metrics

# 导入TFlearn
learn = tf.contrib.learn

# 自定义模型
def my_model(features, target):
    target = tf.one_hot(target, depth=3, on_value=1, off_value=0)

    logits, loss = learn.models.logistic_regression(features, target)

    train_op = tf.contrib.layers.optimize_loss(
        loss,                                      # 损失函数
        tf.contrib.framework.get_global_step(),    # 获取训练步数并且在训练的时候更新
        optimizer='Adagrad',                       # 定义优化器
        learning_rate=0.1                          # 定义学习率
    )

    return tf.argmax(logits, 1), loss, train_op

# 加载iris数据集
iris = datasets.load_iris()
x_train, x_test, y_train, y_test = model_selection.train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)

# 对自定义的模型进行分装
classifier = learn.Estimator(model_fn=my_model)
# 使用封装好的模型和训练数据来执行100次迭代
classifier.fit(x_train, y_train, steps=100)
# 使用训练好的模型来预测
y_predicted = classifier.predict(x_test)
y_predicted = [x for x in y_predicted]
# 计算模型的准确度
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: %2.f%%' % (score *100))


# 本模型主要介绍的是使用tensorflow框架来搭建卷积神经网络，模型主要分为三个部分
## mnist_inference.py 主要介绍的是模型的参数设定和模型的前向传播过程
### 本网络主要的结构就说LeNet-5模型结构：输入层->卷积层->池化层->卷积层->池化层->全连接->全连接->输出层
### 加入了正则化，dropout，以及tf.contrib.layers.xavier_initializer()初始化参数的方法
## mnist_train.py  主要介绍的是模型的训练过程
### 加入的参数的滑动平均以及学习率衰减，并且将训练的模型用tf.Saver给保存起来
### 得到的结果如下：
After 1000 training steps, loss on training batch is 0.126925<br>
After 2000 training steps, loss on training batch is 0.0853739<br>
After 3000 training steps, loss on training batch is 0.0815417<br>
After 4000 training steps, loss on training batch is 0.0907585<br>
After 5000 training steps, loss on training batch is 0.0641611<br>
After 6000 training steps, loss on training batch is 0.0578219<br>
After 7000 training steps, loss on training batch is 0.0681541<br>
After 8000 training steps, loss on training batch is 0.0447801<br>
After 9000 training steps, loss on training batch is 0.0440486<br>
After 10000 training steps, loss on training batch is 0.0515769<br>
......<br>
After 27000 training steps, loss on training batch is 0.034204<br>
After 28000 training steps, loss on training batch is 0.0368325<br>
After 29000 training steps, loss on training batch is 0.0345642<br>
After 30000 training steps, loss on training batch is 0.0363364<br>
## mnist_eval.py  主要用于验证模型的结果，加载训练得到的最终模型，并且用验证数据来验证结果
### 得到的结果如下：
After 30000 training steps validation accuracy = 0.9934<br>
### 从结果可以看出，模型的正确率比起之前的全连接神经网络的98.4%有所提高
## model_mnist  主要保存的是训练好得到的模型，方便验证的时候总结加载
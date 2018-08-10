# 本部分主要介绍了使用神经网络来对著名的MNIST数据集进行分类
## 其中运用了很多之前学过的东西比如学习率衰减，模型参数的滑动平均，正则化等等
### 完整的mnist程序的流程：<br>
1. 加载数据 <br>
2. 设置模型的参数 <br>
3. 定义前向传播函数(分为滑动平均和非滑动平均)<br>
4. 训练过程<br>
   1) 定义输入和输出<br>
   2) 生成隐藏层的参数(初始化)<br>
   3) 利用前向传播函数来计算输出的y(不包含滑动平均)，训练时的输出<br>
   4) 初始化滑动平均类，并且利用前向传播函数来输出average_y， 验证/测试时候的输出<br>
   5) 计算损失函数，这里使用的是sparse_softmax_cross_entropy_with_logits，其对与单分类问题的来说计算的速度快了，<br>
      且这里的使用的label为非稀疏的，在这里使用的计算的过程中，得到的结果的shape=(batch_size, 1)，所以计算的时候要mean操作<br>
   6) 考虑正则化，将之前计算的交叉熵损失函数加上正则项，从而可以得到总的损失<br>
   7) 学习率指数衰减<br>
   8) 定义模型的训练过程，其中利用control_dependencies，来将训练步骤和参数的滑动平均合为一步进行执行<br>
   9) 计算模型的精确度。在计算的过程中，由于y_和y都是使用了one-hot编码之后的数，所以在判断是哪一类的时候需要执行argmax操作，<br>
      同时也需要进行mean操作，目的是取整个batch的平均的正确率<br>
   10) 定义了两种的训练模型的方法。具体为：开启会话/获取数据/迭代循环/sess.run()来启动训练，这里每次循环都使用有个batch<br>
    的大小来进行训练。第二种方法的区别在于，其每次迭代的过程种都使用了batch，但是当全部的数据都过一遍后，才进行下一次循环迭代<br>
    
    注意：在考虑滑动平均参数的时候，一定要考虑到在训练的时候使用的是没有滑动的参数来训练迭代，但是在预测的时候使用的是滑动平均处理后的结果<br>
### 运行的结果如下<br>
After 0 training steps, validation accurary is 0.0638<br>
After 1000 training steps, validation accurary is 0.9788<br>
After 2000 training steps, validation accurary is 0.9826<br>
After 3000 training steps, validation accurary is 0.9842<br>
After 4000 training steps, validation accurary is 0.9858<br>
After 5000 training steps, validation accurary is 0.9848<br>
After 6000 training steps, validation accurary is 0.986<br>
After 7000 training steps, validation accurary is 0.985<br>
After 8000 training steps, validation accurary is 0.9854<br>
After 9000 training steps, validation accurary is 0.9858<br>
After 10000 training steps, validation accurary is 0.9854<br>
After 11000 training steps, validation accurary is 0.9854<br>
After 12000 training steps, validation accurary is 0.9858<br>
After 13000 training steps, validation accurary is 0.9856<br>
After 14000 training steps, validation accurary is 0.9848<br>
After 15000 training steps, validation accurary is 0.9852<br>
After 16000 training steps, validation accurary is 0.9844<br>
After 17000 training steps, validation accurary is 0.9852<br>
After 18000 training steps, validation accurary is 0.9852<br>
After 19000 training steps, validation accurary is 0.985<br>
After 20000 training steps, validation accurary is 0.9848<br>
After 21000 training steps, validation accurary is 0.9852<br>
After 22000 training steps, validation accurary is 0.985<br>
After 23000 training steps, validation accurary is 0.9848<br>
After 24000 training steps, validation accurary is 0.9854<br>
After 25000 training steps, validation accurary is 0.9854<br>
After 26000 training steps, validation accurary is 0.9844<br>
After 27000 training steps, validation accurary is 0.9846<br>
After 28000 training steps, validation accurary is 0.9848<br>
After 29000 training steps, validation accurary is 0.9844<br>
可以看出验证集的精度在不断的提升<br>
### 验证集和测试集之间的差距如下
![](https://github.com/Anosy/tensorflow_DL/blob/master/Mnist/Mnist_1/testvalidate.png)<br>


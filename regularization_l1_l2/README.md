# 本目录主要介绍的就是考虑正则化问题
## 5layerNN+l2.py 主要是利用了正则化，对5层神经网络进行优化
### 无添加的正则化的神经网络的效果：
![](https://github.com/Anosy/tensorflow_DL/tree/master/regularization_l1_l2/NN.png)
显然出现了严重的过拟合
### 添加上正则化的神经网络的效果：
![](https://github.com/Anosy/tensorflow_DL/tree/master/regularization_l1_l2/NN+L2.png)
显然过拟合消除了
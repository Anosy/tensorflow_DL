# 本目录主要介绍的就是考虑正则化问题
## 5layerNN+l2.py 主要是利用了正则化，对5层神经网络进行优化
### 无添加的正则化的神经网络的效果：
![NN](https://github.com/Anosy/tensorflow_DL/blob/master/regularization_l1_l2/NN.png)<br>
显然出现了严重的过拟合<br>
### 添加上正则化的神经网络的效果：
![NN_L2](https://github.com/Anosy/tensorflow_DL/blob/master/regularization_l1_l2/NN%2BL2.png)<br>
显然过拟合消除了<br>
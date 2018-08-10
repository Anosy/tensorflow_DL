# 本文件夹主要分为两个文件，包含了吴恩达课程中的逻辑回归网络的构建，以及使用tensorflow来构建网络
## 在吴恩达课程中设计的逻辑回归网络的运行结果为

Cost after iteration 0: 0.693147
Cost after iteration 100: 0.584508
Cost after iteration 200: 0.466949
Cost after iteration 300: 0.376007
Cost after iteration 400: 0.331463
Cost after iteration 500: 0.303273
Cost after iteration 600: 0.279880
Cost after iteration 700: 0.260042
Cost after iteration 800: 0.242941
Cost after iteration 900: 0.228004
Cost after iteration 1000: 0.214820
Cost after iteration 1100: 0.203078
Cost after iteration 1200: 0.192544
Cost after iteration 1300: 0.183033
Cost after iteration 1400: 0.174399
train accuracy: 97.60765550239235 %
test accuracy: 70.0 %

## 在自己设计的tensorflow版的逻辑回归模型的结果为
After 0 training step(s), cross entropy on all data is 9.69913
After 100 training step(s), cross entropy on all data is 0.0029648263
After 200 training step(s), cross entropy on all data is 0.00047805256
After 300 training step(s), cross entropy on all data is 0.00035899048
After 400 training step(s), cross entropy on all data is 0.00028490118
After 500 training step(s), cross entropy on all data is 0.00023404916
After 600 training step(s), cross entropy on all data is 0.00019704383
After 700 training step(s), cross entropy on all data is 0.00016896293
After 800 training step(s), cross entropy on all data is 0.0001469649
After 900 training step(s), cross entropy on all data is 0.0001293088
train accuracy:  100%
test accuracy:  64%

### 显然上述两个模型都发生了严重的过拟合
# 本部分主要是介绍tensorflow中的Tensorboard的应用

## 第一部分：介绍tensorboard，具体的代码见：intro_TensorBoard.py
核心代码：<br>

        writer = tf.summary.FileWriter('./log', tf.get_default_graph())
        
生成一个写日志的writer, 并且将当前的Tensorflow计算图给写入到Tensorboard中<br>
结果图：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/first_add_tensorboard.png)<br>

## 第二部分：介绍name_scope，variable_scope的区别, 具体代码见：diff_variable_name_scope.py
**结论：**<br>
在name_scope中，tf.get_variable函数不受其影响，如： b = tf.get_variable('b', [1]) 得到的结果为b:0

## 第三部分：介绍运用name_scope命名空间，并且将其运用到tensorboard中，具体代码见：name_scope_TensorBoard.py
结果图：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/second_add_tensorboard.png)<br>
从图中可以看出，input2的节点概括了第一个图中的大部分节点，从而使得得到的结果更加的清晰

## 第四部分: 在Mnist函数中使用tensorboard来查看模型的结构，具体代码见：Mnist_tensorBoard子文件夹(主要修改部分在mnist_train.py)
核心代码：<br>

        # 定义输入x, y_
        with tf.name_scope('input'):
            ...
        # 定义滑动平均
        with tf.name_scope('moving_average'):
            ...
        # 定义损失函数
        with tf.name_scope('loss_function'):
            ...
        # 定义训练过程
        with tf.name_scope('train_step'):
            ...
        # 生成一个写日志的writer
        writer = tf.summary.FileWriter(logdir='./log', graph=tf.get_default_graph())
        
        with tf.Session() as sess:
            ....
            # 配置运行时需要记录的信息
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            # 运行时记录运行信息的proto
            run_metadata = tf.RunMetadata()
            # 将配置信息和记录运行的信息的proto传入运行的过程
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys},
                    options=run_options, run_metadata=run_metadata)
            # 将节点在运行的信息写入到日志文件
            writer.add_run_metadata(run_metadata, 'step%03d' % i)
            
结果图：<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/tensorboard_mnist.png)

## 第五部分：监控指标可视化, 用来监控模型运行过程中，得出的结果的图等等信息，具体代码见summary_tensorboard.py
核心代码：<br>

        tf.summary.histogram(name, var)  # 用来记录张量元素的取值分布，如模型的参数等
        tf.summary.scalar(name, var) # 用来记录标量元素的取值分布，如学习率和损失等
        tf.summary.image('input', image_shaped_input, 10) # 用来记录图片的信息，这里的10表示最多显示10个图片
        merged = tf.summary.merge_all()  # 将所有的写入操作给合并
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, tf.get_default_graph()) # 初始化一个写入的writer
        summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_:ys})  # 文件的写入过程也需要通过sess才能执行
        summary_writer.add_summary(summary, global_step=i)  # 将生成的日志给写入到文件中

**accuarcy结果图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/accuracy.png)<br>
**cross_entropy结果图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/cross_entropy.png)<br>
**layer1_parameter结果图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/layer1_parameter.png)<br>
**image结果图**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/image.png)<br>
**distribution结果图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/distribution.png)<br>
**histogram结果图：**<br>
![](https://github.com/Anosy/tensorflow_DL/blob/master/TensorBoard/result_picture/histogram.png)<br>
**注：本部分的log生成占用磁盘较大，所以没有上传！**<br>




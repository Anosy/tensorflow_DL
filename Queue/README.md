# 本部分主要介绍的是多线程输入数据处理框架
## 第一部分，介绍什么是队列，具体的代码见：intro_queue.py
**核心部分** <br>
1. q = tf.FIFOQueue(2, 'int32')         创建一个先进先出队列<br>
2. init = q.enqueue_many(([0, 10], ))   初始化队列中的元素<br>
3. x = q.dequeue()                      入列操作<br>
4. q_inc = q.enqueue([y])               出列操作<br>

## 第二部分，介绍多线程的停止控制，主要代码见：intro_Coordinator.py
**核心部分** <br>
1. coord.should_stop()                  停止所有线程<br>
2. coord.request_stop()                 通知线程停止<br>
3. coord = tf.train.Coordinator()       生成一个Coordinator实例<br>
4. coord.join()                         等待所有线程的退出<br>

## 第三部分，介绍启动多线程来操作一个队列，具体代码见：intro_QueueRunner.py
**核心部分** <br>
1. q = tf.FIFOQueue(100, 'float')       声明一个先进先出的队列<br>
2. enqueue_op = q.enqueue([tf.random_normal([1])])    出列操作<br>
3. qr = tf.train.QueueRunner(q, [enqueue_op] * 5)     创建多个线程来进行入列的操作，这里使用了5个线程来进行<br>
4. tf.train.add_queue_runner(qr)                      将定义过的QueueRunner加入tensorflow计算图上指定的集合<br>
5. coord = tf.train.Coordinator()                     使用Coordinator来协同启动的线程<br>
6. threads = tf.train.start_queue_runners(sess=sess, coord=coord)     使用QueueRunner时需要明确调用tf.train.start_queue_runners来启动所有进程<br>
7. coord.request_stop()<br>
   coord.join(threads)                                停止所有线程<br> 

## 第四部分，介绍如何使用TFRecord来生成文件，具体代码见：TFRecord_save.py，文件保存地址为：TFRecord_Output

## 第五部分，介绍如何使用队列来读取文件，具体代码见：load_use_queue.py
**核心部分** <br>
1. files = tf.train.match_filenames_once('./TFRecord_Output/data.tfrecords-*')      获取文件列表<br>
2. filename_queue = tf.train.string_input_producer(files, shuffle=False)            创建文件输入队列<br>
3. reader = tf.TFRecordReader()                                                     创建一个reader来读取TFRecord文件中的样例<br>
4. _, serialized_example = reader.read(filename_queue)                              从文件中读出一个样例<br>
5. tf.parse_single_example(...)                                                     解析读入的一个样例<br>
**注： 这里需要使用 tf.local_variables_initializer() 初始化 tf.train.match_filenames_once() 中的变量，否则会报错** <br>

## 第六部分，介绍如何读取文件，并且将数据给转化成batch的格式，具体代码见：batch_queue.py
**核心部分** <br>
1. batch_size = 3                                       定义batch的大小
2. capacity = 1000 + 3 * batch_size                     定义文件队列最多可以存储的样例的个数
3. example_batch, label_batch = tf.train.batch([example, label], batch_size=batch_size, capacity=capacity)   组合样例
**输出的结果与分析：** <br>

    [0 0 1] [0 1 0] <br>
    [1 0 0] [1 0 1] <br>
    
从结果上可以看出读取数据一次为：<br>
example:0 , label:0<br>
example:0 , label:1<br>
example:1 , label:0<br>
example:1 , label:1<br>

## 第七部分，介绍输入数据的处理框架，具体代码见input_process_frame.py
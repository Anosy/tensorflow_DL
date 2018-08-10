# 保存模型
# import tensorflow as tf
#
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess, './model/model1.ckpt')

# 导入模型
# import tensorflow as tf
# 
# v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
# v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
# result = v1 + v2
# 
# saver = tf.train.Saver()
# 
# with tf.Session() as sess:
#     saver.restore(sess, './model/model1.ckpt')
#     print(sess.run(result))

# 直接加载持久化的图，这样就不需要定义计算方法
# import tensorflow as tf
# # 加载图
# saver = tf.train.import_meta_graph('./model/model1.ckpt.meta')
# with tf.Session() as sess:
#       # 加载参数和操作
#     saver.restore(sess, tf.train.latest_checkpoint('./model'))
#     graph = tf.get_default_graph()
#     # 通过张量的名称来获取张良量
#     v1 = graph.get_tensor_by_name("v1:0")
#     v2 = graph.get_tensor_by_name("v2:0")
#     add = graph.get_tensor_by_name('add:0')
#     print(sess.run(v1))
#     print(sess.run(v2))
#     print(sess.run(add))

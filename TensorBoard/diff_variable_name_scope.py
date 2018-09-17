import tensorflow as tf

with tf.variable_scope('foo'):
     # 在命名空间foo下获取变量'bar'
     a = tf.get_variable('bar', [1])
     print(a.name)
     # 输出：foo/bar:0

with tf.variable_scope('bar'):
    # 在命名空间bar下获取变量'bar'
    b = tf.get_variable('bar', [1])
    print(b.name)
    # 输出：bar/bar:0

with tf.name_scope('a'):
    a = tf.Variable([1])
    print(a.name)
    # 输出：a/Variable:0

    b = tf.get_variable('b', [1])
    print(b.name)
    # 输出：b:0
    # 可以看出tf.get_variable函数不受tf.name_scope函数的影响

with tf.name_scope('b'):
    tf.get_variable('b', [1])
    # 输出：ValueError: Variable b already exists, disallowed. Did you mean to set reuse=True in VarScope? Originally defined at:
    # 由于tf.get_variable不受tf.name_scope函数的影响，所以这里相当重复的声明了
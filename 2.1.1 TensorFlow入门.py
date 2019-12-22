'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    : 2019/12/3 10:00
@Author  :Zhangyunjia
@FileName:  2.1.1 TensorFlow入门.py
@Software: PyCharm
'''
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np

# 1.sigmoid示例

# 定义一个图对象
graph = tf.Graph()
# 定义一个会话对象
session = tf.InteractiveSession(graph=graph)
# graph = tf.get_default_graph() #默认计算图

# 定义张量
x = tf.placeholder(shape=[1, 10], dtype=tf.float32, name='x')
W = tf.Variable(tf.random_uniform(shape=[10, 5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 初始化
tf.global_variables_initializer().run()
# print('w: ',w)
# print('b: ',b)

# 执行该图
h_eval = session.run(h, feed_dict={x: np.random.rand(1, 10)})
print('h_eval: ', h_eval)

# 关闭会话，释放会话对象占用的资源
session.close()


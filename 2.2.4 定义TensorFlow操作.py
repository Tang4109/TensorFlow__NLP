'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    : 2019/12/3 15:45
@Author  :Zhangyunjia
@FileName:  2.2.4 定义TensorFlow操作.py
@Software: PyCharm
'''

import tensorflow.compat.v1 as tf

x = tf.constant([[1, 2], [3, 4]], dtype=tf.int32)
y = tf.constant([[4, 3], [3, 2]], dtype=tf.int32)

# 1.比较操作
x_equal_y = tf.equal(x, y, name=None)
print('\nx_equal_y:\n', x_equal_y)

x_less_y = tf.less(x, y, name=None)
print('\nx_less_y:\n', x_less_y)

x_great_equal_y = tf.greater_equal(x, y, name=None)
print('\nx_great_equal_y:\n', x_great_equal_y)

condition = tf.constant([[True, False], [True, False]], dtype=tf.bool)
x_cond_y = tf.where(condition, x, y, name=None)
print('\nx_cond_y:\n', x_cond_y)

# 2.数学运算
x_add_y = tf.add(x, y)
print('\nx_add_y:\n', x_add_y)

x_mul_y = tf.matmul(x, y)
print('\nx_mul_y:\n', x_mul_y)

x = tf.cast(x, dtype=tf.float32)
log_x = tf.log(x)
print('\nlog_x:\n', log_x)

x_sum_1 = tf.reduce_sum(x, axis=[1], keepdims=False)
print('\nx_sum_1:\n', x_sum_1)

x_sum_2 = tf.reduce_sum(x, axis=[0], keepdims=True)
print('\nx_sum_2:\n', x_sum_2)

data = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=tf.float32)
segment_ids = tf.constant([0, 0, 0, 1, 1, 2, 2, 2, 2, 2], dtype=tf.int32)
x_seg_sum = tf.segment_sum(data, segment_ids)
print('\nx_seg_sum:\n', x_seg_sum)

# 3.分散和聚合操作

# Defining the graph and session
graph = tf.Graph()  # Creates a graph
session = tf.InteractiveSession(graph=graph)  # Creates a session

print('\n分散和聚合操作:')
print('\n1-D scatter operation:')
ref = tf.Variable(tf.constant([1, 9, 3, 10, 5], dtype=tf.float32), name='scatter_update')
indices = [1, 3]
updates = tf.constant([2, 4], dtype=tf.float32)
tf_scatter_update = tf.scatter_update(ref, indices, updates, use_locking=None, name=None)
print('\nref:\n', ref)

print('\nn-D scatter operation:')
indices = [[1], [2]]
updates = tf.constant([[1, 1, 1], [2, 2, 2]])
shape = [4, 3]
tf_scatter_nd_1 = tf.scatter_nd(indices, updates, shape, name=None)
print('\ntf_scatter_nd_1:\n', tf_scatter_nd_1)
print('\nsession.run(tf_scatter_nd_1):\n', session.run(tf_scatter_nd_1))

# n-D scatter operation
indices = [[1, 0], [3, 1]]  # 2 x 2
updates = tf.constant([1, 2])  # 2 x 1
shape = [4, 3]  # 2
tf_scatter_nd_2 = tf.scatter_nd(indices, updates, shape, name=None)

print('\nScatter Operation for n-D')
print(session.run(tf_scatter_nd_2))

# 1-D gather operation
params = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
indices = [1, 4]
tf_gather = tf.gather(params, indices, validate_indices=True, name=None)  # => [2,5]
print('\nGather Operation for 1-D')
print(session.run(tf_gather))

# n-D gather operation
params = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=tf.float32)
indices = [[0], [2]]
tf_gather_nd = tf.gather_nd(params, indices, name=None)  # => [[0,0,0],[2,2,2]]
print('\nGather Operation for n-D')
print(session.run(tf_gather_nd))

params = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 222], [3, 3, 3]], dtype=tf.float32)
indices = [[0, 1], [2, 2]]
tf_gather_nd_2 = tf.gather_nd(params, indices, name=None)  # => [[0,0,0],[2,2,2]]
print('\nGather Operation for n-D')
print(session.run(tf_gather_nd_2))

session.close()

# 4.神经网络相关操作



















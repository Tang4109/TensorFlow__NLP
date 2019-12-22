'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    : 2019/12/3 15:08
@Author  :Zhangyunjia
@FileName:  2.2.1 在TensorFlow中定义输入.py
@Software: PyCharm
'''

# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import os

# 定义一个图对象
graph = tf.Graph()
# 定义一个会话对象
session = tf.InteractiveSession(graph=graph)
# graph = tf.get_default_graph() #默认计算图


# 定义一个文件名队列
filenames = ['test%d.txt' % i for i in range(1, 4)]
filename_queue = tf.train.string_input_producer(filenames, capacity=3, shuffle=True, name='string_input_producer')
# 查看文件是否存在
for f in filenames:
    if not tf.gfile.Exists(f):
        raise ValueError('Failed to find file: ' + f)
    else:
        print('File %s found.' % f)

# 定义一个读取器
reader = tf.TextLineReader()
# 读取键-值
key, value = reader.read(filename_queue, name='text_read_op')
# 定义record_defaults
record_defaults = [[-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0], [-1.0]]
# 将读取到的文本行解码为数字列
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = tf.decode_csv(value, record_defaults=record_defaults)

# 把列拼接起来，形成单个张量（特征）
features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10])
# 将张量进行打乱，按批次输出
x = tf.train.shuffle_batch([features], batch_size=3, capacity=5, name='data_batch', min_after_dequeue=1, num_threads=1)

# 启动管道
coord = tf.train.Coordinator()  # 线程管理器
threads = tf.train.start_queue_runners(coord=coord, sess=session)

# 定义张量
W = tf.Variable(tf.random_uniform(shape=[10, 5], minval=-0.1, maxval=0.1, dtype=tf.float32), name='W')
b = tf.Variable(tf.zeros(shape=[5], dtype=tf.float32), name='b')
h = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 初始化
tf.global_variables_initializer().run()

for step in range(5):
    x_eval, h_eval = session.run([x, h])
    print('========== Step %d ==========' % step)
    print('Evaluated data (x)')
    print(x_eval)
    print('Evaluated data (h)')
    print(h_eval)
    print('')

coord.request_stop()
coord.join(threads)
session.close()




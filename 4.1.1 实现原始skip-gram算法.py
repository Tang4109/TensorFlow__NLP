'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/13 21:23
@Author  : Zhangyunjia
@FileName: 4.1.1 实现原始skip-gram算法.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

import tensorflow.compat.v1 as tf

# 定义占位符
batch_size = 10
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])




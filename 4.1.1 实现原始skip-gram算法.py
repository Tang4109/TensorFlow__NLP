'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/13 21:23
@Author  : Zhangyunjia
@FileName: 4.1.1 实现原始skip-gram算法.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

import bz2
import collections
import csv
import math
import pickle
import random
from urllib.request import urlretrieve
import numpy as np
import nltk
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import os
# 用pickle读取中间变量：
from matplotlib import pylab

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)
with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
data_index = 0


def generate_batch_skip_gram(batch_size, window_size):
    global data_index
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * window_size + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    num_samples = 2 * window_size

    for i in range(batch_size // num_samples):
        k = 0
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            k += 1
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Skip-Gram Algorithm
batch_size = 200  # Data points in a single batch
embedding_size = 128  # Dimension of the embedding vector.
window_size = 4  # How many words to consider left and right.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 50
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)
num_sampled = 32  # Number of negative examples to sample.
vocabulary_size = 50000
# Defining Inputs and Outputs
tf.reset_default_graph()

# 定义占位符
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
# 创建常量
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# 定义两个嵌入层
in_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
out_embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

# 定义模型计算

# 对负样本进行采样
negative_samples, _, _ = tf.nn.log_uniform_candidate_sampler(
    train_labels, num_true=1, num_sampled=num_sampled, unique=True, range_max=vocabulary_size)

# 为输入数据、输出数据、负样本进行词向量映射
in_embed = tf.nn.embedding_lookup(in_embeddings, train_dataset)
out_embed = tf.nn.embedding_lookup(out_embeddings, tf.reshape(train_labels, [-1]))
negative_embed = tf.nn.embedding_lookup(out_embeddings, negative_samples)

# 定义正样本的损失函数log(sigma(v_o * v_i^T))
# 按照某个维度求平均值，axis控制维度，默认对所有的元素求平均
loss = tf.reduce_mean(
    tf.log(
        tf.nn.sigmoid(
            #按照某个维度求和，axis控制维度，默认对所有的元素求和
            tf.reduce_sum(
                # tf.diag创建对角矩阵，以便只保留正样本的内积
                tf.diag([1.0 for _ in range(batch_size)]) *
                tf.matmul(out_embed, tf.transpose(in_embed)),
                # axis=0按列求和，axis=1按行求和
                axis=0)
        )
    )
)
# 定义负样本的损失函数log(sigma(-v_o * v_i^T))

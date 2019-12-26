'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/26 17:47
@Author  : Zhangyunjia
@FileName: 3.5 使用TensorFlow实现CBOW算法.py
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


# 更改数据生成过程

def generate_batch_cbow(batch_size, window_size):
    global data_index
    span = 2 * window_size + 1  # [ skip_window target skip_window ]
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # The buffer holds the data contained within the span
    # 双端队列
    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size):
        target = window_size  # target label at the center of the buffer
        target_to_avoid = [window_size]  # we only need to know the words around a given word, not the word itself
        # add selected target to avoid_list for next time
        col_idx = 0
        for j in range(span):
            # ignore the target word when creating the batch
            if j == span // 2:
                continue
            batch[i, col_idx] = buffer[j]
            col_idx += 1
        labels[i, 0] = buffer[target]
        # 双端列表，后面进入，前面挤出
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

# for window_size in [1, 2]:
#     data_index = 0
#     batch, labels = generate_batch_cbow(batch_size=8, window_size=window_size)
#     print('\nwith window_size = %d:' % (window_size))
#     print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
#     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

vocabulary_size=50000
batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
# How many words to consider left and right.
# Skip gram by design does not require to have all the context words in a given step
# However, for CBOW that's a requirement, so we limit the window size
window_size = 2

# We pick a random validation set to sample nearest neighbors
valid_size = 16 # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 50

# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

num_sampled = 32 # Number of negative examples to sample.

tf.reset_default_graph()

# Training input data (target word IDs). Note that it has 2*window_size columns
train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*window_size])
# Training input label data (context word IDs)
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Embedding layer, contains the word embeddings
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0,dtype=tf.float32))

# Softmax Weights and Biases
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                 stddev=0.5 / math.sqrt(embedding_size),dtype=tf.float32))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))


stacked_embedings = None
print('Defining %d embedding lookups representing each word in the context'%(2*window_size))
for i in range(2*window_size):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
    x_size,y_size = embedding_i.get_shape().as_list()
    if stacked_embedings is None:
        stacked_embedings = tf.reshape(embedding_i,[x_size,y_size,1])
    else:
        stacked_embedings = tf.concat(axis=2,values=[stacked_embedings,tf.reshape(embedding_i,[x_size,y_size,1])])

assert stacked_embedings.get_shape().as_list()[2]==2*window_size
print("Stacked embedding size: %s"%stacked_embedings.get_shape().as_list())
mean_embeddings =  tf.reduce_mean(stacked_embedings,2,keepdims=False)
print("Reduced mean embedding size: %s"%mean_embeddings.get_shape().as_list())

loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))





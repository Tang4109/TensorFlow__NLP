'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/31 9:14
@Author  : Zhangyunjia
@FileName: 4.6 文档分类.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk
import pickle
import pdb


with open('../data/data1.pickle', 'rb') as handle:
    data=pickle.load(handle)
with open('../data/reverse_dictionary1.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
with open('../data/dictionary1.pickle', 'rb') as handle:
    dictionary=pickle.load(handle)
with open('../data/count1.pickle', 'rb') as handle:
    count=pickle.load(handle)
with open('../data/test_data1.pickle', 'rb') as handle:
    test_data=pickle.load(handle)

vocabulary_size = 25000

data_index = 0


def generate_batch(data, batch_size, window_size):
    global data_index
    span = 2 * window_size + 1
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    num_samples = 2 * window_size

    for i in range(batch_size):
        target = window_size  # target label at the center of the buffer
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
    assert batch.shape[0] == batch_size and batch.shape[1] == span - 1
    return batch, labels


# for window_size in [4]:
#     data_index = 0
#     batch, labels = generate_batch(data, batch_size=128, window_size=window_size)
#     pdb.set_trace()
#     print(batch)
#     print('\nwith window_size = %d:' % (window_size))
#     print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
#     print('    labels:', [reverse_dictionary[li] for li in labels.reshape(128)])
#
# pdb.set_trace()
test_data_index = 0

def generate_test_batch(data, batch_size):
    global test_data_index
    batch = np.ndarray(shape=(batch_size,), dtype=np.int32)
    for bi in range(batch_size):
        batch[bi] = data[test_data_index]
        test_data_index = (test_data_index + 1) % len(data)

    return batch

# test_data_index = 0
# test_batch = generate_test_batch(test_data[list(test_data.keys())[0]], batch_size=8)
# print('\nwith window_size = %d:' % (window_size))
# print('    labels:', [reverse_dictionary[li] for li in test_batch.reshape(8)])


batch_size = 128
embedding_size = 128
window_size = 4
valid_size = 16
valid_window = 50
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)
num_sampled = 32

tf.reset_default_graph()
train_dataset = tf.placeholder(tf.int32, shape=[batch_size,2*window_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
test_labels = tf.placeholder(tf.int32, shape=[batch_size],name='test_dataset')

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0,dtype=tf.float32))
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                     stddev=1.0 / math.sqrt(embedding_size),dtype=tf.float32))
softmax_biases = tf.Variable(tf.zeros([vocabulary_size],dtype=tf.float32))

mean_batch_embedding = tf.reduce_mean(tf.nn.embedding_lookup(embeddings,test_labels),axis=0)
stacked_embedings = None
print('Defining %d embedding lookups representing each word in the context'%(2*window_size))
for i in range(2*window_size):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
    x_size,y_size = embedding_i.get_shape().as_list()
    if stacked_embedings is None:
        stacked_embedings = tf.reshape(embedding_i,[x_size,y_size,1])
    else:
        stacked_embedings = tf.concat(axis=2,values=[stacked_embedings,tf.reshape(embedding_i,[x_size,y_size,1])])

# Make sure the staked embeddings have 2*window_size columns
assert stacked_embedings.get_shape().as_list()[2]==2*window_size
print("Stacked embedding size: %s"%stacked_embedings.get_shape().as_list())

# Compute mean embeddings by taking the mean of the tensor containing the stack of embeddings
mean_embeddings =  tf.reduce_mean(stacked_embedings,2,keepdims=False)
print("Reduced mean embedding size: %s"%mean_embeddings.get_shape().as_list())

loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                           labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

num_steps = 100001
cbow_loss = []

config=tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True

with tf.Session(config=config) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    average_loss = 0

    for step in range(num_steps):

        batch_data, batch_labels = generate_batch(data, batch_size, window_size)

        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            cbow_loss.append(average_loss)
            average_loss = 0

        if (step + 1) % 10000 == 0:
            sim = similarity.eval()

            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

    num_test_steps = 100

    document_embeddings = {}
    print('Testing Phase (Compute document embeddings)')
    for k, v in test_data.items():
        print('\tCalculating mean embedding for document ', k, ' with ', num_test_steps, ' steps.')
        test_data_index = 0
        topic_mean_batch_embeddings = np.empty((num_test_steps, embedding_size), dtype=np.float32)
        for test_step in range(num_test_steps):
            test_batch_labels = generate_test_batch(v, batch_size)
            batch_mean = session.run(mean_batch_embedding, feed_dict={test_labels: test_batch_labels})
            topic_mean_batch_embeddings[test_step, :] = batch_mean
        document_embeddings[k] = np.mean(topic_mean_batch_embeddings, axis=0)



# 用pickle保存中间变量：
with open('../data/document_embeddings1.pickle', 'wb') as handle:
    pickle.dump(document_embeddings, handle, protocol=2)

kmeans = KMeans(n_clusters=5, random_state=43643, max_iter=10000, n_init=100, algorithm='elkan')
kmeans.fit(np.array(list(document_embeddings.values())))

# Compute items fallen within each cluster
document_classes = {}
for inp, lbl in zip(list(document_embeddings.keys()), kmeans.labels_):
    if lbl not in document_classes:
        document_classes[lbl] = [inp]
    else:
        document_classes[lbl].append(inp)
for k,v in document_classes.items():
    print('\nDocuments in Cluster ',k)
    print('\t',v)




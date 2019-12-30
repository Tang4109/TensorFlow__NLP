'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/30 12:54
@Author  : Zhangyunjia
@FileName: 4.5.2 实现Glove.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

from __future__ import print_function
import collections
import csv
import math
import pickle

import numpy as np
import os
import random
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import bz2
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
#nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil

with open('data.pickle', 'rb') as handle:
    data=pickle.load(handle)
with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
with open('dictionary.pickle', 'rb') as handle:
    dictionary=pickle.load(handle)
with open('count.pickle', 'rb') as handle:
    count=pickle.load(handle)
with open('cooc_mat.pickle', 'rb') as handle:
    cooc_mat=pickle.load(handle)

vocabulary_size = 50000
data_index=0
def generate_batch(batch_size, window_size):
    # data_index is updated by 1 everytime we read a data point
    global data_index

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    weights = np.ndarray(shape=(batch_size), dtype=np.float32)

    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    span = 2 * window_size + 1

    # The buffer holds the data contained within the span
    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # This is the number of context words we sample for a single target word
    num_samples = 2 * window_size

    # We break the batch reading into two for loops
    # The inner for loop fills in the batch and labels with
    # num_samples data points using data contained withing the span
    # The outper for loop repeat this for batch_size//num_samples times
    # to produce a full batch
    for i in range(batch_size // num_samples):
        k = 0
        # avoid the target word itself as a prediction
        # fill in batch and label numpy arrays
        for j in list(range(window_size)) + list(range(window_size + 1, 2 * window_size + 1)):
            batch[i * num_samples + k] = buffer[window_size]
            labels[i * num_samples + k, 0] = buffer[j]
            weights[i * num_samples + k] = abs(1.0 / (j - window_size))
            k += 1

            # Everytime we read num_samples data points,
        # we have created the maximum number of datapoints possible
        # withing a single span, so we need to move the span by 1
        # to create a fresh new span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels, weights

print('Sample chunks of co-occurance matrix')

# Basically calculates the highest cooccurance of several chosen word
for i in range(10):
    idx_target = i

    # get the ith row of the sparse matrix and make it dense
    ith_row = cooc_mat.getrow(idx_target)
    ith_row_dense = ith_row.toarray('C').reshape(-1)

    # select target words only with a reasonable words around it.
    while np.sum(ith_row_dense) < 10 or np.sum(ith_row_dense) > 50000:
        # Choose a random word
        idx_target = np.random.randint(0, vocabulary_size)

        # get the ith row of the sparse matrix and make it dense
        ith_row = cooc_mat.getrow(idx_target)
        ith_row_dense = ith_row.toarray('C').reshape(-1)

    print('\nTarget Word: "%s"' % reverse_dictionary[idx_target])

    sort_indices = np.argsort(ith_row_dense).reshape(-1)  # indices with highest count of ith_row_dense
    sort_indices = np.flip(sort_indices, axis=0)  # reverse the array (to get max values to the start)

    # printing several context words to make sure cooc_mat is correct
    print('Context word:', end='')
    for j in range(10):
        idx_context = sort_indices[j]
        print('"%s"(id:%d,count:%.2f), ' % (reverse_dictionary[idx_context], idx_context, ith_row_dense[idx_context]),
              end='')
    print()


batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4 # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors
valid_size = 16 # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 50
# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)

num_sampled = 32 # Number of negative examples to sample.

epsilon = 1 # used for the stability of log in the loss function

tf.reset_default_graph()

# Training input data (target word IDs).
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# Training input label data (context word IDs)
train_labels = tf.placeholder(tf.int32, shape=[batch_size])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Variables.
in_embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
in_bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='embeddings_bias')

out_embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
out_bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='embeddings_bias')

# Look up embeddings for inputs and outputs
# Have two seperate embedding vector spaces for inputs and outputs
embed_in = tf.nn.embedding_lookup(in_embeddings, train_dataset)
embed_out = tf.nn.embedding_lookup(out_embeddings, train_labels)
embed_bias_in = tf.nn.embedding_lookup(in_bias_embeddings,train_dataset)
embed_bias_out = tf.nn.embedding_lookup(out_bias_embeddings,train_labels)

# weights used in the cost function
weights_x = tf.placeholder(tf.float32,shape=[batch_size],name='weights_x')
# Cooccurence value for that position
x_ij = tf.placeholder(tf.float32,shape=[batch_size],name='x_ij')

loss = tf.reduce_mean(
    weights_x * (tf.reduce_sum(embed_in*embed_out,axis=1) + embed_bias_in + embed_bias_out - tf.log(epsilon+x_ij))**2)

# Compute the similarity between minibatch examples and all embeddings.
# We use the cosine distance:
embeddings = (in_embeddings + out_embeddings)/2.0
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# Optimizer.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

num_steps = 100001
glove_loss = []
skip_window = 4 # How many words to consider left and right.

average_loss = 0
Config=tf.ConfigProto(allow_soft_placement=True)  ##:如果你指定的设备不存在,允许TF自动分配设备
Config.gpu_options.allow_growth=True  ##动态分配内存
with tf.Session(config=Config) as session:
    tf.global_variables_initializer().run()
    print('Initialized')

    for step in range(num_steps):

        # generate a single batch (data,labels,co-occurance weights)
        batch_data, batch_labels, batch_weights = generate_batch(
            batch_size, skip_window)

        # Computing the weights required by the loss function
        batch_weights = []  # weighting used in the loss function
        batch_xij = []  # weighted frequency of finding i near j

        # Compute the weights for each datapoint in the batch
        for inp, lbl in zip(batch_data, batch_labels.reshape(-1)):
            point_weight = (cooc_mat[inp, lbl] / 100.0) ** 0.75 if cooc_mat[inp, lbl] < 100.0 else 1.0
            batch_weights.append(point_weight)
            batch_xij.append(cooc_mat[inp, lbl])
        batch_weights = np.clip(batch_weights, -100, 1)
        batch_xij = np.asarray(batch_xij)

        # Populate the feed_dict and run the optimizer (minimize loss)
        # and compute the loss. Specifically we provide
        # train_dataset/train_labels: training inputs and training labels
        # weights_x: measures the importance of a data point with respect to how much those two words co-occur
        # x_ij: co-occurence matrix value for the row and column denoted by the words in a datapoint
        feed_dict = {train_dataset: batch_data.reshape(-1), train_labels: batch_labels.reshape(-1),
                     weights_x: batch_weights, x_ij: batch_xij}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # Update the average loss variable
        average_loss += l
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            glove_loss.append(average_loss)
            average_loss = 0

        # Here we compute the top_k closest words for a given validation word
        # in terms of the cosine distance
        # We do this for all the words in the validation set
        # Note: This is an expensive step
        if step % 10000 == 0:
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

    final_embeddings = normalized_embeddings.eval()

np.save('Glove_embeddings', final_embeddings)

with open('struct_skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(final_embeddings)


# 绘图
def find_clustered_embeddings(embeddings, distance_threshold, sample_threshold):
    # 计算余弦相似度
    cosine_sim = np.dot(embeddings, np.transpose(embeddings))
    # embeddings ** 2对每个元素进行平方
    # axis=1 按行求和，reshape转回原来的维度
    norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
                  np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
    assert cosine_sim.shape == norm.shape
    # 应该对norm开根号才对
    cosine_sim /= np.sqrt(norm)
    # 将对角线元素填充为-1，排除自身
    np.fill_diagonal(cosine_sim, -1.0)
    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    for _ in range(sample_threshold - 1):
        # 寻找相似度最大的单词的索引
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1

    max_cosine_sim = np.max(mod_cos_sim, axis=1)
    # 这里的意思是选出与10个以上单词余弦距离大于0.25的单词，即寻找紧密聚集的嵌入，返回单词所在行数
    return np.where(max_cosine_sim > distance_threshold)[0]


# 前1000个单词绘图展示
num_points = 1000  # we will use a large sample space to build the T-SNE manifold and then prune it using cosine similarity
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold
selected_embeddings = final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
selected_ids = find_clustered_embeddings(selected_embeddings, .25, 10)
two_d_embeddings = two_d_embeddings[selected_ids, :]

print('Out of ', num_points, ' samples, ', selected_ids.shape[0], ' samples were selected by pruning')


# Plotting the t-SNE Results with Matplotlib
def plot(embeddings, labels):
    n_clusters = 20  # number of clusters
    cmap = pylab.cm.get_cmap("Spectral")
    label_colors = [cmap(float(i) / n_clusters) for i in range(n_clusters)]

    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    # Define K-Means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
    kmeans_labels = kmeans.labels_

    pylab.figure(figsize=(15, 15))  # in inches

    # plot all the embeddings and their corresponding words
    for i, (label, klabel) in enumerate(zip(labels, kmeans_labels)):
        x, y = embeddings[i, :]
        pylab.scatter(x, y, c=label_colors[klabel])

        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom', fontsize=7)

    pylab.show()


words = [reverse_dictionary[i] for i in selected_ids]
plot(two_d_embeddings, words)



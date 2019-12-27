'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/27 16:31
@Author  : Zhangyunjia
@FileName: 4.4.2 结构化skip_gram算法.py
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
from matplotlib import pylab
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

tf.disable_eager_execution()
import os

vocabulary_size=50000
with open('dictionary.pickle', 'rb') as handle:
    dictionary=pickle.load(handle)

with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
with open('count.pickle', 'rb') as handle:
    count=pickle.load(handle)
with open('data.pickle', 'rb') as handle:
    data=pickle.load(handle)


data_index = 0


def generate_batch(batch_size, window_size):
    global data_index

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    # Note that the labels array has 2*window_size columns
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 2 * window_size), dtype=np.int32)

    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # for a full length of batch size, we do the following
    # make the target word the i th input word (i th row of batch)
    # make all the context words the columns of labels array
    # Update the data index and the buffer
    for i in range(batch_size):
        batch[i] = buffer[window_size]
        labels[i, :] = [buffer[span_idx] for span_idx in
                        list(range(0, window_size)) + list(range(window_size + 1, span))]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 2 # How many words to consider left and right.

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

# Training input data (target word IDs).
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# Training input label data (context word IDs)
train_labels = [tf.placeholder(tf.int32, shape=[batch_size, 1]) for _ in range(2*window_size)]
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)


embeddings = tf.Variable(
tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
softmax_weights = [tf.Variable(
tf.truncated_normal([vocabulary_size, embedding_size],
                     stddev=0.5 / math.sqrt(embedding_size))) for _ in range(2*window_size)]
softmax_biases = [tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01)) for _ in range(2*window_size)]

embed = tf.nn.embedding_lookup(embeddings, train_dataset)
loss = tf.reduce_sum(
[
    tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights[wi], biases=softmax_biases[wi], inputs=embed,
                           labels=train_labels[wi], num_sampled=num_sampled, num_classes=vocabulary_size))
    for wi in range(window_size*2)
]
)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

num_steps = 100001
skip_gram_loss = []  # Collect the sequential loss values for plotting purposes

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
            batch_size, window_size)
        feed_dict = {train_dataset: batch_data}
        for wi in range(2 * window_size):
            feed_dict.update({train_labels[wi]: np.reshape(batch_labels[:, wi], (-1, 1))})

        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            skip_gram_loss.append(average_loss)
            average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
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
    skip_gram_final_embeddings = normalized_embeddings.eval()

# We will save the word vectors learned and the loss over time
# as this information is required later for comparisons
np.save('struct_skip_embeddings', skip_gram_final_embeddings)

with open('struct_skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_gram_loss)


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
selected_embeddings = skip_gram_final_embeddings[:num_points, :]
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

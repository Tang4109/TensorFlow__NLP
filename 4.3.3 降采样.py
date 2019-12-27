'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/27 11:18
@Author  : Zhangyunjia
@FileName: 4.3.2 实现基于unigram的负采样.py
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

vocabulary_size = 50000
with open('dictionary.pickle', 'rb') as handle:
    dictionary = pickle.load(handle)

with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)
with open('count.pickle', 'rb') as handle:
    count = pickle.load(handle)
with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)

token_count = len(data)
word_count_dictionary = {}
unigrams = [0 for _ in range(vocabulary_size)]
for word, w_count in count:
    w_idx = dictionary[word]
    unigrams[w_idx] = w_count * 1.0 / token_count
    word_count_dictionary[w_idx] = w_count
print('First 10 Unigram probabilities')
print(unigrams[:10])

subsampled_data = []
drop_count = 0
drop_examples = []

# 降采样
for w_i in data:
    p_w_i = 1 - np.sqrt(1e5 / word_count_dictionary[w_i])

    if np.random.random() < p_w_i:
        drop_count += 1
        drop_examples.append(reverse_dictionary[w_i])
    else:
        subsampled_data.append(w_i)

data_index = 0



def generate_batch_cbow(batch_size, window_size):
    # window_size is the amount of words we're looking at from each side of a given word
    # creates a single batch
    # data_index is updated by 1 everytime we read a set of data point
    global data_index

    # span defines the total window size, where
    # data we consider at an instance looks as follows.
    # [ skip_window target skip_window ]
    # e.g if skip_window = 2 then span = 5
    span = 2 * window_size + 1  # [ skip_window target skip_window ]

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    # Note that batch has span-1=2*window_size columns
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # The buffer holds the data contained within the span
    buffer = collections.deque(maxlen=span)

    # Fill the buffer and update the data_index
    for _ in range(span):
        buffer.append(subsampled_data[data_index])
        data_index = (data_index + 1) % len(subsampled_data)

    # Here we do the batch reading
    # We iterate through each batch index
    # For each batch index, we iterate through span elements
    # to fill in the columns of batch array
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

        # Everytime we read a data point,
        # we need to move the span by 1
        # to create a fresh new span
        buffer.append(subsampled_data[data_index])
        data_index = (data_index + 1) % len(subsampled_data)

    return batch, labels


batch_size = 128  # Data points in a single batch
embedding_size = 128  # Dimension of the embedding vector.
# How many words to consider left and right.
# Skip gram by design does not require to have all the context words in a given step
# However, for CBOW that's a requirement, so we limit the window size
window_size = 2

# We pick a random validation set to sample nearest neighbors
valid_size = 16  # Random set of words to evaluate similarity on.
# We sample valid datapoints randomly from a large window without always being deterministic
valid_window = 50

# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)

num_sampled = 32  # Number of negative examples to sample.

tf.reset_default_graph()

# Training input data (target word IDs). Note that it has 2*window_size columns
train_dataset = tf.placeholder(tf.int32, shape=[batch_size, 2 * window_size])
# Training input label data (context word IDs)
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Embedding layer, contains the word embeddings
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))

# Softmax Weights and Biases
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=0.5 / math.sqrt(embedding_size), dtype=tf.float32))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))

stacked_embedings = None
print('Defining %d embedding lookups representing each word in the context' % (2 * window_size))
for i in range(2 * window_size):
    embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:, i])
    x_size, y_size = embedding_i.get_shape().as_list()
    if stacked_embedings is None:
        stacked_embedings = tf.reshape(embedding_i, [x_size, y_size, 1])
    else:
        stacked_embedings = tf.concat(axis=2, values=[stacked_embedings, tf.reshape(embedding_i, [x_size, y_size, 1])])

assert stacked_embedings.get_shape().as_list()[2] == 2 * window_size
print("Stacked embedding size: %s" % stacked_embedings.get_shape().as_list())
mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())

candidate_sampler = tf.nn.fixed_unigram_candidate_sampler(true_classes=tf.cast(train_labels, dtype=tf.int64),
                                                          num_true=1,
                                                          num_sampled=num_sampled,
                                                          unique=True, range_max=vocabulary_size,
                                                          distortion=0.75,
                                                          num_reserved_ids=0,
                                                          unigrams=unigrams, name='unigram_sampler')

loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size,
                               sampled_values=candidate_sampler))

optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

num_steps = 100001
cbow_loss_unigram_subsampled = []
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # Initialize the variables in the graph
    tf.global_variables_initializer().run()
    print('Initialized')

    average_loss = 0

    # Train the Word2vec model for num_step iterations
    for step in range(num_steps):

        # Generate a single batch of data
        batch_data, batch_labels = generate_batch_cbow(batch_size, window_size)

        # Populate the feed_dict and run the optimizer (minimize loss)
        # and compute the loss
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)

        # Update the average loss variable
        average_loss += l

        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
            cbow_loss_unigram_subsampled.append(average_loss)
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0

        # Evaluating validation set word similarities
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            # Here we compute the top_k closest words for a given validation word
            # in terms of the cosine distance
            # We do this for all the words in the validation set
            # Note: This is an expensive step
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    cbow_final_embeddings = normalized_embeddings.eval()

with open('cbow_loss_unigram_subsampled.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(cbow_loss_unigram_subsampled)


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
selected_embeddings = cbow_final_embeddings[:num_points, :]
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

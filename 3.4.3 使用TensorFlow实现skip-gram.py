'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/13 13:36
@Author  : Zhangyunjia
@FileName: 3.4.3 使用TensorFlow实现skip-gram.py
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
#用pickle读取中间变量：
from matplotlib import pylab

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

with open('data.pickle', 'rb') as handle:
    data=pickle.load(handle)
with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary=pickle.load(handle)
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
batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4 # How many words to consider left and right.
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 50
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(range(1000, 1000+valid_window), valid_size),axis=0)
num_sampled = 32 # Number of negative examples to sample.
vocabulary_size=50000
# Defining Inputs and Outputs
tf.reset_default_graph()
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
softmax_weights = tf.Variable(
    tf.truncated_normal([vocabulary_size, embedding_size],
                        stddev=0.5 / math.sqrt(embedding_size)))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))
# Defining the Model Computations
embed = tf.nn.embedding_lookup(embeddings, train_dataset)
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(
        weights=softmax_weights, biases=softmax_biases, inputs=embed,
        labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
# 计算余弦相似性
#reduce_sum-压缩求和，用于降维
#norm等于行向量的模
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
#单位化：向量除以模得到单位向量
normalized_embeddings = embeddings / norm
#选择验证集
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
#验证集中每个词与所有词的余弦相似性
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# Optimizer.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
num_steps = 100001
skip_losses = []
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # Initialize the variables in the graph
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        # Generate a single batch of data
        batch_data, batch_labels = generate_batch_skip_gram(
            batch_size, window_size)

        # Populate the feed_dict and run the optimizer (minimize loss)
        # and compute the loss
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        # Update the average loss variable
        average_loss += l
        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000

            skip_losses.append(average_loss)
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0

        # Evaluating validation set word similarities
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                #因为argsort从小到大寻找索引，所以sim前面添加负号，即寻找相似性最大的8个词的位置索引
                #从1开始是为了排除自己和自己的相似性
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    skip_gram_final_embeddings = normalized_embeddings.eval()

np.save('skip_embeddings', skip_gram_final_embeddings)

with open('skip_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_losses)
# Visulizing the Learnings of the Skip-Gram Algorithm
def find_clustered_embeddings(embeddings, distance_threshold, sample_threshold):
    #计算余弦相似度
    cosine_sim = np.dot(embeddings, np.transpose(embeddings))
    #embeddings ** 2对每个元素进行平方
    #axis=1 按行求和，reshape转回原来的维度
    norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
                  np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
    assert cosine_sim.shape == norm.shape
    #应该对norm开根号才对
    cosine_sim /= np.sqrt(norm)
    #将对角线元素填充为-1，排除自身
    np.fill_diagonal(cosine_sim, -1.0)
    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    for _ in range(sample_threshold - 1):
        #寻找相似度最大的单词的索引
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1

    max_cosine_sim = np.max(mod_cos_sim, axis=1)
    #这里的意思是选出与10个以上单词余弦距离大于0.25的单词，即寻找紧密聚集的嵌入，返回单词所在行数
    return np.where(max_cosine_sim > distance_threshold)[0]
#前1000个单词绘图展示
num_points = 1000 # we will use a large sample space to build the T-SNE manifold and then prune it using cosine similarity
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold
selected_embeddings = skip_gram_final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
selected_ids = find_clustered_embeddings(selected_embeddings,.25,10)
two_d_embeddings = two_d_embeddings[selected_ids,:]

print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')
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
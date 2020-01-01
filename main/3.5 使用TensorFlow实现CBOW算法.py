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
import pdb
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

pdb.set_trace()
for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_cbow(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' % (window_size))
    print('    batch:', [[reverse_dictionary[bii] for bii in bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

vocabulary_size = 50000
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

# 定义嵌入层（词向量）
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))

# 定义Softmax 权重和偏置
softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                  stddev=0.5 / math.sqrt(embedding_size), dtype=tf.float32))
softmax_biases = tf.Variable(tf.random_uniform([vocabulary_size], 0.0, 0.01))

# 使用降维运算符，通过平均最后一个轴上的堆叠嵌入，将矩阵维度减小到[batch_size,embedding_size]
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

# 对上下文单词求平均，转化为一个矩阵
mean_embeddings = tf.reduce_mean(stacked_embedings, 2, keepdims=False)
print("Reduced mean embedding size: %s" % mean_embeddings.get_shape().as_list())

# 定义损失函数
loss = tf.reduce_mean(
    tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=mean_embeddings,
                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

# 定义优化函数
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# 使用余弦距离计算minibatch示例与所有嵌入之间的相似度
# norm=模长
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
# 单位化unitization
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
    normalized_embeddings, valid_dataset)
# 计算验证集和所有嵌入之间的相似度
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# 运行算法
num_steps = 100001  # 迭代次数
cbow_loss = []

# 执行图形
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # Initialize the variables in the graph
    tf.global_variables_initializer().run()
    print('Initialized')

    average_loss = 0
    # 循环训练num_steps次
    for step in range(num_steps):
        # 获取批训练数据
        batch_data, batch_labels = generate_batch_cbow(batch_size, window_size)
        # 填充feed_dict并运行优化器(最小化损失)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        # 计算损失
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        # 更新平均损失变量
        average_loss += l

        # 显示训练过程
        if (step + 1) % 2000 == 0:
            # 每训练2000次展示一次
            if step > 0:
                average_loss = average_loss / 2000
            cbow_loss.append(average_loss)
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            average_loss = 0

        # 每训练10000次评估一次验证集单词相似性
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            # 根据余弦距离计算给定验证词的top_k最接近的单词
            # 注意:这是一个昂贵的步骤
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

#保存最终目标词嵌入
np.save('cbow_embeddings',cbow_final_embeddings)
#保存损失变量以便绘图
with open('cbow_loss.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(cbow_loss)

#绘图
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
selected_embeddings = cbow_final_embeddings[:num_points, :]
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
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
batch_size = 128
# Data points in a single batch
embedding_size = 128  # Dimension of the embedding vector.
window_size = 4  # How many words to consider left and right.
valid_size = 16  # Random set of words to evaluate similarity on.
valid_window = 50
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size), axis=0)
num_sampled = 32  # Number of negative examples to sample.
vocabulary_size = 5000
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
loss1 = tf.reduce_mean(
    tf.log(
        tf.clip_by_value(
            tf.nn.sigmoid(
                # 按照某个维度求和，axis控制维度，默认对所有的元素求和
                tf.reduce_sum(
                    # tf.diag创建对角矩阵，以便只保留正样本的内积
                    tf.diag([1.0 for _ in range(batch_size)]) *
                    tf.matmul(out_embed, tf.transpose(in_embed)),
                    # axis=0按列求和，axis=1按行求和
                    axis=0)
            )
        ,1e-8,1.0)
    )
)

# 定义负样本的损失函数log(sigma(-v_o * v_i^T))\
loss2 = tf.reduce_mean(
    tf.reduce_sum(
        tf.log(
            tf.clip_by_value(
                tf.nn.sigmoid(
                    -tf.matmul(
                        negative_embed, tf.transpose(in_embed)
                    )
                )
            ,1e-8,1.0)
    )
        , axis=0
    )
)

# 将对数似然转换为负对数似然
loss = -1.0*loss1-loss2

# 使用余弦距离计算minibatch示例与所有嵌入之间的相似度
# norm=每一行的模
norm = tf.sqrt(
    tf.reduce_sum(
        tf.square(
            out_embeddings
        )
        , 1, keep_dims=True)
)
# 按行单位化
normalized_embeddings = out_embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

# 定义一个恒定的学习率和一个使用Adagrad方法的优化器
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

# 运行原始的skip-gram算法
# 运行次数
num_steps = 100001
# 收集连续损失值以进行绘图
skip_gram_loss_original = []

# ConfigProto是一种提供各种配置设置的方法
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # 初始化图中变量
    tf.global_variables_initializer().run()
    print('Initialized...')
    average_loss = 0
    # 训练Word2vec模型进行num_step次迭代
    for step in range(num_steps):
        # 生成单批数据
        batch_data, batch_labels = generate_batch_skip_gram(batch_size, window_size)
        # 填充feed_dict
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        # 运行优化器（使损失最小化）并计算损失
        _, loss_ = session.run([optimizer, loss], feed_dict=feed_dict)
        # 更新平均损失变量
        average_loss = average_loss + loss_
        # 最近2000批次中的平均损失
        if (step + 1) % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            print(negative_embed)
            print('Average loss at step %d: %f' % (step + 1, average_loss))

            # 将损失存入list中以便画图
            skip_gram_loss_original.append(average_loss)
            # 损失变量归0，以便下次使用
            average_loss = 0
        # 每隔10000次，根据余弦距离计算给定验证字的top_k最近字
        # 对验证集中的所有单词执行此操作，注意：这是一个昂贵的步骤
        if (step + 1) % 10000 == 0:
            sim = similarity.eval()
            print(sim)
            for i in range(valid_size):
                # 索引转单词
                valid_word = reverse_dictionary[valid_examples[i]]
                # 最近邻居的数量
                top_k = 8
                # 因为argsort从小到大寻找索引，所以sim前面添加负号，即寻找相似性最大的8个词的位置索引
                # 从1开始是为了排除自己和自己的相似性
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    # 存储最终的词向量
    skip_gram_original_final_embeddings = normalized_embeddings.eval()
# 保存到本地
np.save('skip_gram_original_embeddings', skip_gram_original_final_embeddings)
# 保存原始skip-gram损失，以便绘图
with open('skip_original_losses.csv', 'wt') as f:
    # delimiter-定界符
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_gram_loss_original)
# 绘制改进的Skip-Gram损失与原始Skip-Gram损失的关系图
skip_loss_path = os.path.join('skip_losses.csv')
# 加载改进的skip-gram损失的数据
with open(skip_loss_path, 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for r_i, row in enumerate(reader):
        if r_i == 0:
            skip_gram_loss = [float(s) for s in row]
# 定义画板
pylab.figure(figsize=(15, 5))
# 定义x轴
x = np.arange(len(skip_gram_loss)) * 2000
# 绘图
pylab.plot(x, skip_gram_loss, label="Skip-Gram (Improved)", linestyle='--', linewidth=2)
pylab.plot(x, skip_gram_loss_original, label="Skip-Gram (Original)", linewidth=2)
# 设置图形属性
pylab.title('Original vs Improved Skip-Gram Loss Decrease Over Time', fontsize=24)
pylab.xlabel('Iterations', fontsize=22)
pylab.ylabel('Loss', fontsize=22)
pylab.legend(loc=1, fontsize=22)
# 保存图形
pylab.savefig('loss_skipgram_original_vs_improve.jpg')
pylab.show()

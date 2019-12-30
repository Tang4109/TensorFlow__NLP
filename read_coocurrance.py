'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/30 15:16
@Author  : Zhangyunjia
@FileName: read_coocurrance.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

from __future__ import print_function
import collections
import math
import pickle

import numpy as np
import os
import random
import tensorflow as tf
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

data_index = 0

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

vocabulary_size = 50000
cooc_data_index = 0
dataset_size = len(data) # We iterate through the full text
skip_window = 4 # How many words to consider left and right.

# The sparse matrix that stores the word co-occurences
cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)
print(cooc_mat.shape)


def generate_cooc(batch_size, skip_window):
    '''
    Generate co-occurence matrix by processing batches of data
    '''
    data_index = 0
    print('Running %d iterations to compute the co-occurance matrix' % (dataset_size // batch_size))
    for i in range(dataset_size // batch_size):
        # Printing progress
        if i > 0 and i % 100000 == 0:
            print('\tFinished %d iterations' % i)

        # Generating a single batch of data
        batch, labels, weights = generate_batch(batch_size, skip_window)
        labels = labels.reshape(-1)

        # Incrementing the sparse matrix entries accordingly
        for inp, lbl, w in zip(batch, labels, weights):
            cooc_mat[inp, lbl] += (1.0 * w)

generate_cooc(8,skip_window)
# 用pickle保存中间变量：
with open('cooc_mat.pickle', 'wb') as handle:
    pickle.dump(cooc_mat, handle, protocol=2)
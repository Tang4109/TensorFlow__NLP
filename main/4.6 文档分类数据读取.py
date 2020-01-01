'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/31 9:39
@Author  : Zhangyunjia
@FileName: 4.6 文档分类数据读取.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import nltk
import pickle
import pdb
url = 'http://mlg.ucd.ie/files/datasets/'


def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename


filename = maybe_download('../data/bbc-fulltext.zip', 2874078)


def read_data(filename):
    data = []
    files_to_read_for_topic = 250
    topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    with zipfile.ZipFile(filename) as z:
        parent_dir = z.namelist()[0]
        for t in topics:
            print('\tFinished reading data for topic: ', t)
            for fi in range(1, files_to_read_for_topic):
                #03d表示三位数右对齐，0填充
                with z.open(parent_dir + t + '/' + format(fi, '03d') + '.txt') as f:
                    file_string = f.read().decode('latin-1')
                    file_string = file_string.lower()
                    file_string = nltk.word_tokenize(file_string)
                    data.extend(file_string)

        return data


def read_test_data(filename):
    test_data = {}
    files_to_read_for_topic = 250
    topics = ['business', 'entertainment', 'politics', 'sport', 'tech']
    with zipfile.ZipFile(filename) as z:
        parent_dir = z.namelist()[0]
        for t in topics:
            print('\tFinished reading data for topic: ', t)

            for fi in np.random.randint(1, files_to_read_for_topic, (10)).tolist():
                with z.open(parent_dir + t + '/' + format(fi, '03d') + '.txt') as f:
                    file_string = f.read().decode('latin-1')
                    file_string = file_string.lower()
                    file_string = nltk.word_tokenize(file_string)
                    test_data[t + '-' + str(fi)] = file_string

        return test_data


print('Processing training data...')
words = read_data(filename)
print('\nProcessing testing data...')
pdb.set_trace()
test_words = read_test_data(filename)

print('Example words (start): ', words[:10])
print('Example words (end): ', words[-10:])

vocabulary_size = 25000


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    assert len(dictionary) == vocabulary_size
    return data, count, dictionary, reverse_dictionary

def build_dataset_with_existing_dictionary(words, dictionary):
    data = list()
    for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
        data.append(index)
    return data

data1, count1, dictionary1, reverse_dictionary1 = build_dataset(words)

test_data1 = {}
for k, v in test_words.items():
    print('Building Test Dataset for ', k, ' topic')
    test_data1[k] = build_dataset_with_existing_dictionary(v, dictionary1)

print('Most common words (+UNK)', count1[:5])
print('Sample data', data1[:10])
print('test keys: ', test_data1.keys())
del words  # Hint to reduce memory.
del test_words

with open('../data/data1.pickle', 'wb') as handle:
    pickle.dump(data1, handle, protocol=2)
with open('../data/count1.pickle', 'wb') as handle:
    pickle.dump(count1, handle, protocol=2)
with open('../data/dictionary1.pickle', 'wb') as handle:
    pickle.dump(dictionary1, handle, protocol=2)
with open('../data/reverse_dictionary1.pickle', 'wb') as handle:
    pickle.dump(reverse_dictionary1, handle, protocol=2)
with open('../data/test_data1.pickle', 'wb') as handle:
    pickle.dump(test_data1, handle, protocol=2)
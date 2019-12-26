'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/21 20:15
@Author  : Zhangyunjia
@FileName: readData.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
import bz2
import collections
import pickle
from urllib.request import urlretrieve

import nltk
import tensorflow.compat.v1 as tf
import os


# 1.准备数据
def maybe_download(url, filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, filename)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

url = 'http://www.evanjones.ca/software/'
filename = maybe_download(url,'wikipedia2text-extracted.txt.bz2', 18377035,force=False)


def read_data(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with bz2.BZ2File(filename) as f:
        data = []
        file_string = f.read().decode('utf-8')
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
    return data

words = read_data(filename)
vocabulary_size = 5000
def build_dataset(words):
    count = [['UNK', -1]]
    # Gets only the vocabulary_size most common words as the vocabulary
    # All the other words will be replaced with UNK token
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()

    # Create an ID for each word by giving the current length of the dictionary
    # And adding that item to the dictionary
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    # Traverse through all the text we have and produce a list
    # where each element corresponds to the ID of the word found at that index
    for word in words:
        # If word is in the dictionary use the word ID,
        # else use the ID of the special token "UNK"
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)

    # update the count variable with the number of UNK occurences
    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # Make sure the dictionary is of size of the vocabulary
    assert len(dictionary) == vocabulary_size

    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.

#用pickle保存中间变量：
with open('data.pickle', 'wb') as handle:
    pickle.dump(data, handle, protocol=2)
with open('count.pickle', 'wb') as handle:
    pickle.dump(count, handle, protocol=2)
with open('dictionary.pickle', 'wb') as handle:
    pickle.dump(dictionary, handle, protocol=2)
with open('reverse_dictionary.pickle', 'wb') as handle:
    pickle.dump(reverse_dictionary, handle, protocol=2)
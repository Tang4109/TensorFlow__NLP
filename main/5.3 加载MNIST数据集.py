'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2020/1/2 10:40
@Author  : Zhangyunjia
@FileName: 5.3 加载MNIST数据集.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''
import gzip
import os
import pickle
import struct
from urllib.request import urlretrieve
import numpy as np
import pdb
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


def read_mnist(fname_img, fname_lbl, one_hot=False):
    print('\nReading files %s and %s' % (fname_img, fname_lbl))

    # Processing images
    with gzip.open(fname_img) as fimg:
        #magic它是一个文件协议的描述
        #>指大端(用来定义字节是如何存储的)
        #I是指一个无符号整数
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        print(num, rows, cols)

        img = (np.frombuffer(fimg.read(num * rows * cols), dtype=np.uint8).reshape(num, rows, cols, 1)).astype(
            np.float32)
        print('(Images) Returned a tensor of shape ', img.shape)

        # img = (img - np.mean(img)) /np.std(img)
        img *= 1.0 / 255.0

    # Processing labels
    with gzip.open(fname_lbl) as flbl:
        # flbl.read(8) reads upto 8 bytes
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.frombuffer(flbl.read(num), dtype=np.int8)
        if one_hot:
            one_hot_lbl = np.zeros(shape=(num, 10), dtype=np.float32)
            one_hot_lbl[np.arange(num), lbl] = 1.0
        print('(Labels) Returned a tensor of shape: %s' % lbl.shape)
        print('Sample labels: ', lbl[:10])

    if not one_hot:
        return img, lbl
    else:
        return img, one_hot_lbl


# Download data if needed

url = 'http://yann.lecun.com/exdb/mnist/'
# training data
maybe_download(url,'../data/train-images-idx3-ubyte.gz',9912422)
maybe_download(url,'../data/train-labels-idx1-ubyte.gz',28881)
# testing data
maybe_download(url,'../data/t10k-images-idx3-ubyte.gz',1648877)
maybe_download(url,'../data/t10k-labels-idx1-ubyte.gz',4542)
# Read the training and testing data
# pdb.set_trace()
train_inputs, train_labels = read_mnist('../data/train-images-idx3-ubyte.gz',
                                        '../data/train-labels-idx1-ubyte.gz',True)
# pdb.set_trace()
test_inputs, test_labels = read_mnist('../data/t10k-images-idx3-ubyte.gz',
                                      '../data/t10k-labels-idx1-ubyte.gz',True)

with open('../data/train_inputs.pickle', 'wb') as handle:
    pickle.dump(train_inputs, handle, protocol=2)
with open('../data/train_labels.pickle', 'wb') as handle:
    pickle.dump(train_labels, handle, protocol=2)

with open('../data/test_inputs.pickle', 'wb') as handle:
    pickle.dump(test_inputs, handle, protocol=2)
with open('../data/test_labels.pickle', 'wb') as handle:
    pickle.dump(test_labels, handle, protocol=2)
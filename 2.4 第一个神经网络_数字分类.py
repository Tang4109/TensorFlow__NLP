'''
!/usr/bin/env python
 _*_coding:utf-8 _*_
@Time    : 2019/12/4 14:58
@Author  :Zhangyunjia
@FileName:  2.4 第一个神经网络_数字分类.py
@Software: PyCharm
'''

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

import numpy as np
import struct
import gzip
import os
from six.moves.urllib.request import urlretrieve
import matplotlib.pyplot as plt

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


def read_mnist(fname_img, fname_lbl):
    print('\nReading files %s and %s' % (fname_img, fname_lbl))

    with gzip.open(fname_img) as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        print(num, rows, cols)
        img = (np.frombuffer(fimg.read(num * rows * cols), dtype=np.uint8).reshape(num, rows * cols)).astype(np.float32)
        print('(Images) Returned a tensor of shape ', img.shape)

        img = (img - np.mean(img)) / np.std(img)

    with gzip.open(fname_lbl) as flbl:
        # flbl.read(8) reads upto 8 bytes
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.frombuffer(flbl.read(num), dtype=np.int8)
        print('(Labels) Returned a tensor of shape: %s' % lbl.shape)
        print('Sample labels: ', lbl[:10])

    return img, lbl


# Download data if needed
url = 'http://yann.lecun.com/exdb/mnist/'
# training data
maybe_download(url, 'train-images-idx3-ubyte.gz', 9912422)
maybe_download(url, 'train-labels-idx1-ubyte.gz', 28881)
# testing data
maybe_download(url, 't10k-images-idx3-ubyte.gz', 1648877)
maybe_download(url, 't10k-labels-idx1-ubyte.gz', 4542)

# Read the training and testing data
train_inputs, train_labels = read_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_inputs, test_labels = read_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')

# 2.定义TensorFlow图


WEIGHTS_STRING = 'weights'
BIAS_STRING = 'bias'

batch_size = 100

img_width, img_height = 28, 28
input_size = img_height * img_width
num_labels = 10
# Defining inputs and outputs
tf_inputs = tf.placeholder(shape=[batch_size, input_size], dtype=tf.float32, name='inputs')
tf_labels = tf.placeholder(shape=[batch_size, num_labels], dtype=tf.float32, name='labels')


# Defining the Tensorflow variables
def define_net_parameters():
    with tf.variable_scope('layer1'):

        tf.get_variable(WEIGHTS_STRING, shape=[input_size, 500],
                        initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[500],
                        initializer=tf.random_uniform_initializer(0, 0.01))

    with tf.variable_scope('layer2'):
        tf.get_variable(WEIGHTS_STRING, shape=[500, 250],
                        initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[250],
                        initializer=tf.random_uniform_initializer(0, 0.01))

    with tf.variable_scope('output'):
        tf.get_variable(WEIGHTS_STRING, shape=[250, 10],
                        initializer=tf.random_normal_initializer(0, 0.02))
        tf.get_variable(BIAS_STRING, shape=[10],
                        initializer=tf.random_uniform_initializer(0, 0.01))

define_net_parameters()
# Defining calcutations in the neural network starting from inputs to logits
# logits are the values before applying softmax to the final output

def inference(x):
    # calculations for layer 1
    with tf.variable_scope('layer1', reuse=True):
        w, b = tf.get_variable(WEIGHTS_STRING), tf.get_variable(BIAS_STRING)
        tf_h1 = tf.nn.relu(tf.matmul(x, w) + b, name='hidden1')

    # calculations for layer 2
    with tf.variable_scope('layer2', reuse=True):
        w, b = tf.get_variable(WEIGHTS_STRING), tf.get_variable(BIAS_STRING)
        tf_h2 = tf.nn.relu(tf.matmul(tf_h1, w) + b, name='hidden1')

    # calculations for output layer
    with tf.variable_scope('output', reuse=True):
        w, b = tf.get_variable(WEIGHTS_STRING), tf.get_variable(BIAS_STRING)
        tf_logits = tf.nn.bias_add(tf.matmul(tf_h2, w), b, name='logits')

    return tf_logits


# defining the loss
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=inference(tf_inputs), labels=tf_labels))

# defining the optimize function
tf_loss_minimize = tf.train.MomentumOptimizer(momentum=0.9, learning_rate=0.01).minimize(tf_loss)

# defining predictions
tf_predictions = tf.nn.softmax(inference(tf_inputs))


graph = tf.get_default_graph()

session = tf.InteractiveSession(graph=graph)

tf.global_variables_initializer().run()

NUM_EPOCHS = 50


def accuracy(predictions, labels):
    ''' Measure the classification accuracy of some predictions (softmax outputs)
    and labels (integer class labels)'''
    return np.sum(np.argmax(predictions, axis=1).flatten() == labels.flatten()) / batch_size


test_accuracy_over_time = []
train_loss_over_time = []

for epoch in range(NUM_EPOCHS):
    train_loss = []

    # Training Phase
    for step in range(train_inputs.shape[0] // batch_size):
        # Creating one-hot encoded labels with labels
        # One-hot encoding dight 3 for 10-class MNIST data set will result in
        # [0,0,0,1,0,0,0,0,0,0]
        labels_one_hot = np.zeros((batch_size, num_labels), dtype=np.float32)
        labels_one_hot[np.arange(batch_size), train_labels[step * batch_size:(step + 1) * batch_size]] = 1.0

        # Printing the one-hot labels
        if epoch == 0 and step == 0:
            print('Sample labels (one-hot)')
            print(labels_one_hot[:10])
            print()

        # Running the optimization process
        loss, _ = session.run([tf_loss, tf_loss_minimize], feed_dict={
            tf_inputs: train_inputs[step * batch_size: (step + 1) * batch_size, :],
            tf_labels: labels_one_hot})
        train_loss.append(loss)  # Used to average the loss for a single epoch

    test_accuracy = []
    # Testing Phase
    for step in range(test_inputs.shape[0] // batch_size):
        test_predictions = session.run(tf_predictions, feed_dict={
            tf_inputs: test_inputs[step * batch_size: (step + 1) * batch_size, :]})
        batch_test_accuracy = accuracy(test_predictions, test_labels[step * batch_size: (step + 1) * batch_size])
        test_accuracy.append(batch_test_accuracy)

    print('Average train loss for the %d epoch: %.3f\n' % (epoch + 1, np.mean(train_loss)))
    train_loss_over_time.append(np.mean(train_loss))
    print('\tAverage test accuracy for the %d epoch: %.2f\n' % (epoch + 1, np.mean(test_accuracy) * 100.0))
    test_accuracy_over_time.append(np.mean(test_accuracy) * 100)

print('//////////////////////////////////')
print([v.name for v in tf.global_variables()])
session.close()

x_axis = np.arange(len(train_loss_over_time))
# fig = plt.figure() #Here is your error
# ax = fig.add_subplot(1,2)
fig, ax = plt.subplots(nrows=1, ncols=2)
fig.set_size_inches(w=25,h=5)
ax[0].plot(x_axis, train_loss_over_time)
ax[0].set_xlabel('Epochs',fontsize=18)
ax[0].set_ylabel('Average train loss',fontsize=18)
ax[0].set_title('Training Loss over Time',fontsize=20)
ax[1].plot(x_axis, test_accuracy_over_time)
ax[1].set_xlabel('Epochs',fontsize=18)
ax[1].set_ylabel('Test accuracy',fontsize=18)
ax[1].set_title('Test Accuracy over Time',fontsize=20)
fig.savefig('mnist_stats.jpg')
plt.show()






























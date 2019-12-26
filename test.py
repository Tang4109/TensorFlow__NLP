'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/24 16:28
@Author  : Zhangyunjia
@FileName: test.py
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


url = 'http://www.evanjones.ca/software/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
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

filename = maybe_download('wikipedia2text-extracted.txt.bz2', 18377035)

def read_data(filename):
  """
  Extract the first file enclosed in a zip file as a list of words
  and pre-processes it using the nltk python library
  """

  with bz2.BZ2File(filename) as f:

    data = []
    file_size = os.stat(filename).st_size
    chunk_size = 1024 * 1024 # reading 1 MB at a time as the dataset is moderately large
    print('Reading data...')
    for i in range(math.ceil(file_size // chunk_size) + 1):
        bytes_to_read = min(chunk_size,file_size-(i*chunk_size))
        file_string = f.read(bytes_to_read).decode('utf-8')
        file_string = file_string.lower()
        # tokenizes a string to words residing in a list
        file_string = nltk.word_tokenize(file_string)
        data.extend(file_string)
  return data

words = read_data(filename)
print('Data size %d' % len(words))
token_count = len(words)

print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])

# we restrict our vocabulary size to 50000
vocabulary_size = 50000


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

data_index = 0


def generate_batch_skip_gram(batch_size, window_size):
    # data_index is updated by 1 everytime we read a data point
    global data_index

    # two numpy arras to hold target words (batch)
    # and context words (labels)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

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
            k += 1

            # Everytime we read num_samples data points,
        # we have created the maximum number of datapoints possible
        # withing a single span, so we need to move the span by 1
        # to create a fresh new span
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


print('data:', [reverse_dictionary[di] for di in data[:8]])

for window_size in [1, 2]:
    data_index = 0
    batch, labels = generate_batch_skip_gram(batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' % window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


#%%

batch_size = 128 # Data points in a single batch
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4 # How many words to consider left and right.

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
train_labels = tf.placeholder(tf.int64, shape=[batch_size, 1])
# Validation input data, we don't need a placeholder
# as we have already defined the IDs of the words selected
# as validation data
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

#%% md

### Defining Model Parameters and Other Variables

#%%

# Variables

# Embedding layers, contains the word embeddings
# We define two embedding layers
# in_embeddings is used to lookup embeddings corresponding to target words (inputs)
in_embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
)
# out_embeddings is used to lookup embeddings corresponding to contect words (labels)
out_embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
)


# 1. Compute negative sampels for a given batch of data
# Returns a [num_sampled] size Tensor
negative_samples, _, _ = tf.nn.log_uniform_candidate_sampler(train_labels, num_true=1, num_sampled=num_sampled,
                                                           unique=True, range_max=vocabulary_size)
# 2. Look up embeddings for inputs, outputs and negative samples.
in_embed = tf.nn.embedding_lookup(in_embeddings, train_dataset)
out_embed = tf.nn.embedding_lookup(out_embeddings, tf.reshape(train_labels,[-1]))
negative_embed = tf.nn.embedding_lookup(out_embeddings, negative_samples)

# 3. Manually defining negative sample loss
# As Tensorflow have a limited amount of flexibility in the built-in sampled_softmax_loss function,
# we have to manually define the loss fuction.

# 3.1. Computing the loss for the positive sample
# Exactly we compute log(sigma(v_o * v_i^T)) with this equation
loss = tf.reduce_mean(
  tf.log(
      tf.nn.sigmoid(
          tf.reduce_sum(
              tf.diag([1.0 for _ in range(batch_size)])*
              tf.matmul(out_embed,tf.transpose(in_embed)),
          axis=0)
      )
  )
)

# 3.2. Computing loss for the negative samples
# We compute sum(log(sigma(-v_no * v_i^T))) with the following
# Note: The exact way this part is computed in TensorFlow library appears to be
# by taking only the weights corresponding to true samples and negative samples
# and then computing the softmax_cross_entropy_with_logits for that subset of weights.
# More infor at: https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/nn_impl.py
# Though the approach is different, the idea remains the same
loss += tf.reduce_mean(
  tf.reduce_sum(
      tf.log(tf.nn.sigmoid(-tf.matmul(negative_embed,tf.transpose(in_embed)))),
      axis=0
  )
)

# The above is the log likelihood.
# We would like to transform this to the negative log likelihood
# to convert this to a loss. This provides us with
# L = - (log(sigma(v_o * v_i^T))+sum(log(sigma(-v_no * v_i^T))))
loss *= -1.0


# Compute the similarity between minibatch examples and all embeddings.
# We use the cosine distance:
norm = tf.sqrt(tf.reduce_sum(tf.square((in_embeddings+out_embeddings)/2.0), 1, keepdims=True))
normalized_embeddings = out_embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(
normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# Optimizer.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

num_steps = 100001
skip_gram_loss_original = []  # Collect the sequential loss values for plotting purposes

# ConfigProto is a way of providing various configuration settings
# required to execute the graph
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    # Initialize the variables in the graph
    tf.global_variables_initializer().run()
    print('Initialized')
    average_loss = 0

    # Train the Word2vec model for num_step iterations
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
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step + 1, average_loss))
            skip_gram_loss_original.append(average_loss)
            average_loss = 0

        # Here we compute the top_k closest words for a given validation word
        # in terms of the cosine distance
        # We do this for all the words in the validation set
        # Note: This is an expensive step
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

    skip_gram_original_final_embeddings = normalized_embeddings.eval()
np.save('skip_original_embeddings', skip_gram_original_final_embeddings)

with open('skip_original_losses.csv', 'wt') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(skip_gram_loss_original)

# Load the skip-gram losses from the calculations we did in Chapter 3
# So you need to make sure you have this csv file before running the code below
skip_loss_path = os.path.join('..', 'ch3', 'skip_losses.csv')
with open(skip_loss_path, 'rt') as f:
    reader = csv.reader(f, delimiter=',')
    for r_i, row in enumerate(reader):
        if r_i == 0:
            skip_gram_loss = [float(s) for s in row]

pylab.figure(figsize=(15, 5))  # figure in inches

# Define the x axis
x = np.arange(len(skip_gram_loss)) * 2000

# Plot the skip_gram_loss (loaded from chapter 3)
pylab.plot(x, skip_gram_loss, label="Skip-Gram (Improved)", linestyle='--', linewidth=2)
# Plot the original skip gram loss from what we just ran
pylab.plot(x, skip_gram_loss_original, label="Skip-Gram (Original)", linewidth=2)

# Set some text around the plot
pylab.title('Original vs Improved Skip-Gram Loss Decrease Over Time', fontsize=24)
pylab.xlabel('Iterations', fontsize=22)
pylab.ylabel('Loss', fontsize=22)
pylab.legend(loc=1, fontsize=22)

# use for saving the figure if needed
pylab.savefig('loss_skipgram_original_vs_impr.jpg')
pylab.show()














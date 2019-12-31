'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/30 22:58
@Author  : Zhangyunjia
@FileName: Plot.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

import pickle
import numpy as np
import os

from matplotlib import pylab
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


with open('data.pickle', 'rb') as handle:
    data = pickle.load(handle)
with open('reverse_dictionary.pickle', 'rb') as handle:
    reverse_dictionary = pickle.load(handle)

def find_clustered_embeddings(embeddings, distance_threshold, sample_threshold):
    '''
    Find only the closely clustered embeddings.
    This gets rid of more sparsly distributed word embeddings and make the visualization clearer
    This is useful for t-SNE visualization

    distance_threshold: maximum distance between two points to qualify as neighbors
    sample_threshold: number of neighbors required to be considered a cluster
    '''

    # calculate cosine similarity
    cosine_sim = np.dot(embeddings, np.transpose(embeddings))
    norm = np.dot(np.sum(embeddings ** 2, axis=1).reshape(-1, 1),
                  np.sum(np.transpose(embeddings) ** 2, axis=0).reshape(1, -1))
    assert cosine_sim.shape == norm.shape
    cosine_sim /= norm

    # make all the diagonal entries zero otherwise this will be picked as highest
    np.fill_diagonal(cosine_sim, -1.0)

    argmax_cos_sim = np.argmax(cosine_sim, axis=1)
    mod_cos_sim = cosine_sim
    # find the maximums in a loop to count if there are more than n items above threshold
    for _ in range(sample_threshold - 1):
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]), argmax_cos_sim] = -1

    max_cosine_sim = np.max(mod_cos_sim, axis=1)

    return np.where(max_cosine_sim > distance_threshold)[0]



# Load the previously saved embeddings from Chapter 3 exercise
emb_path = os.path.join('Glove_embeddings.npy')
final_embeddings = np.load(emb_path)
num_points = 1000 # we will use a large sample space to build the T-SNE manifold and then prune it using cosine similarity

# Create a t-SNE object from scikit-learn
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
print('Fitting embeddings to T-SNE')
# Get the T-SNE manifold for embeddings
selected_embeddings = final_embeddings[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# Prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
# Prune skip-gram
selected_ids = find_clustered_embeddings(selected_embeddings,.3,10)
two_d_embeddings = two_d_embeddings[selected_ids,:]

# Some stats about pruning
print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')

def plot_embeddings(embeddings, labels):
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

    pylab.savefig('Glove.png')
    pylab.show()


# Run the function
words = [reverse_dictionary[i] for i in selected_ids]
plot_embeddings(two_d_embeddings, words)


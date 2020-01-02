'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2020/1/1 17:22
@Author  : Zhangyunjia
@FileName: 4.6 文档分类绘图.py
@Software: PyCharm
# @Github: https://github.com/Tang4109
'''

from __future__ import print_function

import pickle
import numpy as np
import os
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE

# emb_path = os.path.join('../model/document_embeddings.npy')
# document_embeddings = np.load(emb_path,allow_pickle=True)
with open('../data/document_embeddings1.pickle', 'rb') as handle:
    document_embeddings=pickle.load(handle)

num_points = 1000
# Create a t-SNE object
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
print('Fitting embeddings to T-SNE')
# get the T-SNE manifold
doc_ids, doc_embeddings = zip(*document_embeddings.items())
two_d_embeddings = tsne.fit_transform(doc_embeddings)
print('\tDone')


def plot(embeddings, labels):
    n_clusters = 5  # number of clusters
    # automatically build a discrete set of colors, each for cluster
    cmap = pylab.cm.get_cmap("Spectral")
    label_colors = [cmap(float(i) / n_clusters) for i in range(n_clusters)]
    label_markers = ['o', '^', 'd', 's', 'x']
    # make sure the number of document embeddings is same as
    # point labels provided
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    pylab.figure(figsize=(15, 15))  # in inches


    def get_label_id_from_key(key):
        '''
        We assign each different category a cluster_id
        This is assigned based on what is contained in the point label
        Not the actual clustering results
        '''
        if 'business' in key:
            return 0
        elif 'entertainment' in key:
            return 1
        elif 'politics' in key:
            return 2
        elif 'sport' in key:
            return 3
        elif 'tech' in key:
            return 4

    # Plot all the document embeddings and their corresponding words
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y, c=label_colors[get_label_id_from_key(label)], s=25,
                      marker=label_markers[get_label_id_from_key(label)])

        # Annotate each point on the scatter plot
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom', fontsize=10)

    # Set plot title
    pylab.title('Document Embeddings visualized with t-SNE', fontsize=24)

    # Use for saving the figure if needed
    pylab.savefig('document_embeddings.png')
    pylab.show()


# Run the plotting function
plot(two_d_embeddings, doc_ids)
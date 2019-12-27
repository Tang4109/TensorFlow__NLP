'''
!/usr/bin/env python
_*_coding: utf-8 _*_
@Time    : 2019/12/27 10:54
@Author  : Zhangyunjia
@FileName: 4.2.1 CBOW和skip_gram的t_SNE可视化.py
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
skip_emb_path = os.path.join('skip_embeddings.npy')
cbow_emb_path = os.path.join('cbow_embeddings.npy')

skip_gram_final_embeddings = np.load(skip_emb_path)
cbow_final_embeddings = np.load(cbow_emb_path)

num_points = 1000 # we will use a large sample space to build the T-SNE manifold and then prune it using cosine similarity

# Create a t-SNE object from scikit-learn
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('Fitting embeddings to T-SNE (skip-gram and CBOW)')
# Get the T-SNE manifold for skip-gram embeddings
print('\tSkip-gram')
sg_selected_embeddings = skip_gram_final_embeddings[:num_points, :]
sg_two_d_embeddings = tsne.fit_transform(sg_selected_embeddings)

# Get the T-SNE manifold for CBOW embeddings
print('\tCBOW')
cbow_selected_embeddings = cbow_final_embeddings[:num_points, :]
cbow_two_d_embeddings = tsne.fit_transform(cbow_selected_embeddings)

print('Pruning the T-SNE embeddings (skip-gram and CBOW)')
# Prune the embeddings by getting ones only more than n-many sample above the similarity threshold
# this unclutters the visualization
# Prune skip-gram
print('\tSkip-gram')
sg_selected_ids = find_clustered_embeddings(sg_selected_embeddings,.3,10)
sg_two_d_embeddings = sg_two_d_embeddings[sg_selected_ids,:]
# Prune CBOW
print('\tCBOW')
cbow_selected_ids = find_clustered_embeddings(cbow_selected_embeddings,.3,10)
cbow_two_d_embeddings = cbow_two_d_embeddings[cbow_selected_ids,:]

# Some stats about pruning
print('Out of ',num_points,' samples (skip-gram), ', sg_selected_ids.shape[0],' samples were selected by pruning')
print('Out of ',num_points,' samples (CBOW), ', cbow_selected_ids.shape[0],' samples were selected by pruning')


def plot_embeddings_side_by_side(sg_embeddings, cbow_embeddings, sg_labels, cbow_labels):
    ''' Plots word embeddings of skip-gram and CBOW side by side as subplots
    '''
    # number of clusters for each word embedding
    # clustering is used to assign different colors as a visual aid
    n_clusters = 20

    # automatically build a discrete set of colors, each for cluster
    print('Define Label colors for %d', n_clusters)
    cmap = pylab.cm.get_cmap("Spectral")
    label_colors = [cmap(float(i) / n_clusters) for i in range(n_clusters)]

    # Make sure number of embeddings and their labels are the same
    assert sg_embeddings.shape[0] >= len(sg_labels), 'More labels than embeddings'
    assert cbow_embeddings.shape[0] >= len(cbow_labels), 'More labels than embeddings'

    print('Running K-Means for skip-gram')
    # Define K-Means
    sg_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(sg_embeddings)
    sg_kmeans_labels = sg_kmeans.labels_
    sg_cluster_centroids = sg_kmeans.cluster_centers_

    print('Running K-Means for CBOW')
    cbow_kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(cbow_embeddings)
    cbow_kmeans_labels = cbow_kmeans.labels_
    cbow_cluster_centroids = cbow_kmeans.cluster_centers_

    print('K-Means ran successfully')

    print('Plotting results')
    pylab.figure(figsize=(25, 20))  # in inches

    # Get the first subplot
    pylab.subplot(1, 2, 1)

    # Plot all the embeddings and their corresponding words for skip-gram
    for i, (label, klabel) in enumerate(zip(sg_labels, sg_kmeans_labels)):
        center = sg_cluster_centroids[klabel, :]
        x, y = cbow_embeddings[i, :]

        # This is just to spread the data points around a bit
        # So that the labels are clearer
        # We repel datapoints from the cluster centroid
        if x < center[0]:
            x += -abs(np.random.normal(scale=2.0))
        else:
            x += abs(np.random.normal(scale=2.0))

        if y < center[1]:
            y += -abs(np.random.normal(scale=2.0))
        else:
            y += abs(np.random.normal(scale=2.0))

        pylab.scatter(x, y, c=label_colors[klabel])
        x = x if np.random.random() < 0.5 else x + 10
        y = y if np.random.random() < 0.5 else y - 10
        pylab.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                       ha='right', va='bottom', fontsize=16)
    pylab.title('t-SNE for Skip-Gram', fontsize=24)

    # Get the second subplot
    pylab.subplot(1, 2, 2)

    # Plot all the embeddings and their corresponding words for CBOW
    for i, (label, klabel) in enumerate(zip(cbow_labels, cbow_kmeans_labels)):
        center = cbow_cluster_centroids[klabel, :]
        x, y = cbow_embeddings[i, :]

        # This is just to spread the data points around a bit
        # So that the labels are clearer
        # We repel datapoints from the cluster centroid
        if x < center[0]:
            x += -abs(np.random.normal(scale=2.0))
        else:
            x += abs(np.random.normal(scale=2.0))

        if y < center[1]:
            y += -abs(np.random.normal(scale=2.0))
        else:
            y += abs(np.random.normal(scale=2.0))

        pylab.scatter(x, y, c=label_colors[klabel])
        x = x if np.random.random() < 0.5 else x + np.random.randint(0, 10)
        y = y + np.random.randint(0, 5) if np.random.random() < 0.5 else y - np.random.randint(0, 5)
        pylab.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points',
                       ha='right', va='bottom', fontsize=16)

    pylab.title('t-SNE for CBOW', fontsize=24)
    # use for saving the figure if needed
    pylab.savefig('tsne_skip_vs_cbow.png')
    pylab.show()


# Run the function
sg_words = [reverse_dictionary[i] for i in sg_selected_ids]
cbow_words = [reverse_dictionary[i] for i in cbow_selected_ids]
plot_embeddings_side_by_side(sg_two_d_embeddings, cbow_two_d_embeddings, sg_words, cbow_words)


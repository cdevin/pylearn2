import numpy as np
from collections import Counter

def map_words(word_dict, clusters):
    """
    When using brown clustering, map id of each word
    to a samller dictionary for each cluster
    """

    # words dictionary
    word_dict = word_dict['unique_words']

    # cluster mapping for the words
    num_clusters = len(np.unique(clusters.values()))
    invalid = num_clusters + 1

    # cluster labels for each word in dictionary
    labels = np.zeros(len(word_dict))
    for i, item in enumerate(word_dict):
        try:
            labels[i] = clusters[item]
        except KeyError:
            labels[i] = invalid

    # remap labels
    counter = Counter(labels)
    max_size = np.max(counter.values())
    print "maximum cluster size is: {}".format(max_size)

    # new word id's
    mapped_dict = np.zeros(len(word_dict))
    for i in xrange(num_clusters + 1):
        cluster_words = np.where(labels == i)[0]
        for j, item in enumerate(cluster_words):
            mapped_dict[item] = j

    return mapped_dict, labels, counter

def BrownClusterDict(cluster_file):
    """
    cluster_file: path to output of clustering done by:
        https://github.com/percyliang/brown-cluster
    sav_path: path to save the clustering labels pkl file
    """

    cluster = open(cluster_file).readlines()
    classes = list(np.unique([item.split()[0] for item in cluster]))
    word_dict = {}
    for item in cluster:
        splitted = item.split()
        word_dict[splitted[1]] = classes.index(splitted[0])

    return word_dict


import os
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.space import VectorSpace
from pylearn2.utils import serial

class PennTree(dense_design_matrix.DenseDesignMatrix):

    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len, brown = None):

        if which_set not in self.valid_set_names:
            raise ValueError("which_set should have one of these values: {}".format(self.valid_set_names))
        data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "PennTreebankCorpus/{}.pkl".format(which_set)))
        data = data['words']
        tot_len = len(data) - seq_len
        x = np.zeros((tot_len, seq_len))
        y = np.zeros((tot_len, 1))
        for i in xrange(tot_len):
            x[i] = data[i:i+seq_len]
            y[i] = data[i + seq_len]

        super(PennTree, self).__init__(X = x, y = y)

        self.brown = brown
        if brown is not None:
            data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "PennTreebankCorpus/{}_brown_{}.pkl".format(which_set, brown)))
            cls = np.zeros((tot_len, 1))
            for i in xrange(tot_len):
                cls[i] = data[i + seq_len]

            source = ('features', 'targets', 'classes')
            space = self.data_specs[0]
            space.components.append(VectorSpace(dim=brown))
            self.data_specs = (space, source)
            self.cls = cls


    def get_data(self):
        if self.brown is None:
            return (self.X, self.y)
        else:
            return (self.X, self.y, self.cls)

def BrownCluster(which_set, cluster_path, save_path):

    """
    Return a dictionary, with cluster labels for each example

    which_set: train, valid, test
    cluster_path: path to the picke file made by BrownClusterDict
    save_path: path to save results
    """

    data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "PennTreebankCorpus/{}.pkl".format(which_set)))
    data = data['words']

    word_dict = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "PennTreebankCorpus/dictionaries.npz"))
    word_dict = word_dict['unique_words']
    cluster = serial.load(cluster_path)
    num_clusters = len(np.unique(cluster.values()))
    invalid = num_clusters + 1
    labels = np.zeros(len(data))
    for i, item in enumerate(data):
        try:
            labels[i] = cluster[word_dict[item]]
        except KeyError:
            labels[i] = invalid

    serial.save("{}{}_brown_{}.pkl".format(save_path, which_set, invalid), labels)

def BrownClusterDict(cluster_file, save_path):
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

    serial.save(save_path, word_dict)


if __name__ == "__main__":

    # read and save clusering results as pkl
    BrownClusterDict('/data/lisa/exp/mirzamom/brown-cluster/train-c50-p1.out/paths', 'bc.pkl')
    # save cluster labels for eah label
    BrownCluster('train', 'bc.pkl', './')
    dd = PennTree('train', 3, brown = 51)

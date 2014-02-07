import os
import numpy as np
from collections import Counter
from pylearn2.datasets import dense_design_matrix
from pylearn2.space import VectorSpace
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from noisylearn.projects.tiled.brown_utils import map_words, BrownClusterDict


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
        y = np.zeros((tot_len, 1), dtype = 'uint32')
        for i in xrange(tot_len):
            x[i] = data[i:i+seq_len]
            y[i] = data[i + seq_len]

        super(PennTree, self).__init__(X = x, y = y)

        self.brown = brown
        if brown is not None:
            cluster_path = "${PYLEARN2_DATA_PATH}/"\
                    "PennTreebankCorpus/brown-cluster/" + \
                    "train-c{}-p1.out/paths".format(brown)
            clusters = BrownClusterDict(preprocess(cluster_path))
            word_dicts = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "PennTreebankCorpus/dictionaries.npz"))
            mapped_dict, clusters = map_words(word_dicts, clusters)

            cls = np.zeros((tot_len, 1))
            for i in xrange(tot_len):
                cls[i] = clusters[y[i]]
                y[i] = mapped_dict[y[i]]

            source = ('features', 'targets', 'classes')
            space = self.data_specs[0]
            space.components.append(VectorSpace(dim=1))
            self.data_specs = (space, source)
            self.cls = cls


    def get_data(self):
        if self.brown is None:
            return (self.X, self.y)
        else:
            return (self.X, self.y, self.cls)


if __name__ == "__main__":
    dd = PennTree('train', 3, brown = 50)


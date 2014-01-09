import os
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class PennTree(dense_design_matrix.DenseDesignMatrix):


    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len):

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

if __name__ == "__main__":

    dd = PennTree('train', 3)

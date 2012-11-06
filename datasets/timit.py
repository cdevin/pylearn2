import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load


class TIMIT(dense_design_matrix.DenseDesignMatrix):
    """
    Pylearn2 wrapper for the Toronto Face Dataset.
    http://aclab.ca/users/josh/TFD.html
    """

    mapper = {'train' : 0, 'valid': 1, 'test': 2}

    def __init__(self, which_set, center = False,
                 shuffle=False, rng=None, seed=132987):

        assert which_set in self.mapper.keys()

        path = "${PYLEARN2_DATA_PATH}/timit/"
        if which_set == 'train':
            x_path = path + 'train_pca_x.npy'
            y_path = path + 'train_fy.npy'
        elif which_set == 'valid':
            x_path = path + 'valid_pca_x.npy'
            y_path = path + 'valid_fy.npy'
        else:
            assert which_set == 'test'
            x_path = path + 'test_pca_x.npy'
            y_path = path + 'test_pca_x.npy'


        data_x = load(x_path)
        data_y = load(y_path)

        m = data_x.shape[0]
        if which_set == 'train':
            assert m == 978900
        elif which_set == 'valid':
            assert m == 145656
        elif which_set == 'test':
            assert m == 57919
        else:
            assert False

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]

        # create view converting for retrieving topological view
        #view_converter = dense_design_matrix.DefaultViewConverter((384))

        # init the super class
        super(TIMIT, self).__init__(X = data_x, y = data_y)

        assert not np.any(np.isnan(self.X))



if __name__ == "__main__":

    TIMIT('train')

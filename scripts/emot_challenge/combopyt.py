import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils import serial

class ComboDataset(dense_design_matrix.DenseDesignMatrix):
    def __init__(self, path = None, start = None, stop = None, shuffle=False,
                rng = None, seed = 132987, center = False,
                scale = False,
                axes=('b', 0, 1, 'c'), preprocessor=None,
                which_ds = 'original'):


        path = preprocess(path)
        file_n = "{}data.h5".format(path)
        if ois.path.isfile(file_n):
            make_new = False
        else:
            maken_new = True

        if make_new:
            self.make_data


        if path is None:
            path = '/data/lisa/data/faces/EmotiW/preproc/'
        data_x = np.memmap(path + 'all_x.npy', mode='r', dtype='float32')
        data_y = np.memmap(path + 'all_y.npy', mode='r', dtype='uint8')
        data_x = data_x[::3]
        data_y = data_y[::3]
        data_x = data_x.reshape((data_y.shape[0], 48 * 48))

        one_hot = np.zeros((len(data_y),7), dtype=np.float32)
        one_hot[np.asarray(range(len(data_y))), data_y] = 1.

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]


        if start is not None or stop is not None:
            if start is None:
                start = 0
            else:
                assert start >= 0
            if stop is None:
                stop = -1
            if stop != -1:
                assert stop > start
            data_x = data_x[start:stop]
            data_y = data_y[start:stop]

        self.axes = axes
        view_converter = dense_design_matrix.DefaultViewConverter((48, 48, 1), axes)
        super(ComboDataset, self).__init__(X=data_x, y=data_y, view_converter=view_converter)
        assert not np.any(np.isnan(self.X))

        if preprocessor is not None:
            preprocessor.apply(self)


    @staticmethod
    def make_data(path, shuffle = True):
        file_n = "{}data.h5".format(path)

        orig_path = '/data/lisa/data/faces/EmotiW/preproc/'
        data_x = np.memmap(orig_path + 'all_x.npy', mode='r', dtype='float32')
        data_y = np.memmap(orig_path + 'all_y.npy', mode='r', dtype='uint8')
        data_x = data_x[::3]
        data_y = data_y[::3]
        data_x = data_x.reshape((data_y.shape[0], 48 * 48))

        one_hot = np.zeros((len(data_y),7), dtype=np.float32)
        one_hot[np.asarray(range(len(data_y))), data_y] = 1.
        data_y = one_hot

        if shuffle:
            rng = rng if rng else np.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]


        h5file, node = ComboDataset.init_hdf5(file_n, (data_x.shape[0], data_x.shape[1]), (data_y.shape[0], 7))


if __name__ == "__main__":
    ComboDataset()

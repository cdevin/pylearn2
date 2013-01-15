import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.tfd import TFD
from pylearn2.utils.serial import load
from noisy_encoder.scripts.datasets.utils import reflect, corner_shuffle

class GoogleTFD(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, image_size = 48,
                 example_range = None, center = False,
                 shuffle=False, rng=None, seed=132987):

        # load tfd
        tfd_train = TFD('train', fold = 0)
        tfd_valid = TFD('valid', fold = 0)
        tfd_test = TFD('test', fold = 0)

        tfd_x = numpy.concatenate((tfd_train.X, tfd_valid.X, tfd_test.X))
        tfd_y = numpy.concatenate((tfd_train.y, tfd_valid.y, tfd_test.y))
        one_hot = numpy.zeros((tfd_y.shape[0], 7), dtype='float32')
        for i in xrange(tfd_y.shape[0]):
            one_hot[i, tfd_y[i]] = 1.
        tfd_y = one_hot

        # load google
        path = '${PYLEARN2_DATA_PATH}/faces/GoogleDataset/Clean/googleDataset_21744_p2.pkl'
        data = load(path)
        google_x = numpy.vstack(data[2]).astype('float32')
        google_y = numpy.vstack(data[3]).astype('float32')

        # mix
        data_x = numpy.concatenate((tfd_x, google_x))
        data_y = numpy.concatenate((tfd_y, google_y))

        if center:
            data_x -= 127.5

        if shuffle:
            rng = rng if rng else numpy.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1))

        # init the super class
        super(GoogleTFD, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))

class GoogleTFDAug(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, image_size = 48,
                 example_range = None, center = False,
                 shuffle=False, rng=None, seed=132987):

        rng = numpy.random.RandomState(seed)

        # load tfd
        tfd_train = TFD('train', fold = 0)
        tfd_valid = TFD('valid', fold = 0)
        tfd_test = TFD('test', fold = 0)

        tfd_x = numpy.concatenate((tfd_train.X, tfd_valid.X, tfd_test.X))
        tfd_y = numpy.concatenate((tfd_train.y, tfd_valid.y, tfd_test.y))
        one_hot = numpy.zeros((tfd_y.shape[0], 7), dtype='float32')
        for i in xrange(tfd_y.shape[0]):
            one_hot[i, tfd_y[i]] = 1.
        tfd_y = one_hot

        # augment tfd
        tfd_x, tfd_y = reflect(tfd_x, tfd_y, (tfd_x.shape[0], 48, 48))
        tfd_x, tfd_y = corner_shuffle(tfd_x, tfd_y, (tfd_x.shape[0], 48, 48), 1, rng)
        #tfd_x, tfd_y = shuffle(tfd_x, tfd_y, rng)


        # load google
        path = '${PYLEARN2_DATA_PATH}/faces/GoogleDataset/Clean/googleDataset_35014_preprocessed.pkl'
        data = load(path)
        google_x = numpy.vstack(data[2]).astype('float32')
        google_y = numpy.vstack(data[3]).astype('float32')

        # mix
        data_x = numpy.concatenate((tfd_x, google_x))
        data_y = numpy.concatenate((tfd_y, google_y))

        if center:
            data_x -= 127.5

        if shuffle:
            rng = rng if rng else numpy.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1))

        # init the super class
        super(GoogleTFDAug, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        assert not numpy.any(numpy.isnan(self.X))



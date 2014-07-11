import os, cPickle, logging
from pylearn2.utils import serial
_logger = logging.getLogger(__name__)

import numpy as np
N = np
from pylearn2.datasets import dense_design_matrix

class Google(dense_design_matrix.DenseDesignMatrix):

    mapper = {'train' : 0, 'test': 0}

    def __init__(self, which_set, center = False, rescale = False, gcn = None,
            one_hot = False, start = None, stop = None, axes=('b', 0, 1, 'c'),
            rng = None, seed = 2322, preprocessor = None):

        assert which_set in self.mapper.keys()

        self.axes = axes

        # we define here:
        ntrain = 28709
        ntest  = 7178

        # we also expose the following details:
        self.img_shape = (1,48,48)
        self.img_size = N.prod(self.img_shape)
        self.n_classes = 7
        self.label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog']

        # load data
        path = "/data/lisa/data/faces/GoogleDataset/Clean/googleDataset_ICML.pkl"
        data = serial.load(path)
        X = data[0]
        y = data[1]

        # shuffle data
        rng = rng if rng else np.random.RandomState(seed)
        rand_idx = rng.permutation(len(X))
        X = X[rand_idx]
        y = y[rand_idx]

        if which_set == 'train':
            X = X[:ntrain]
            y = y[:ntrain]
        else:
            X = X[ntrain:]
            y = y[ntrain:]

        view_converter = dense_design_matrix.DefaultViewConverter((48,48,1), axes)


        super(Google,self).__init__(X = X, y = y, view_converter = view_converter)

        assert not np.any(np.isnan(self.X))

        if preprocessor:
             preprocessor.apply(self)


    def get_test_set(self):
        return Google(which_set='test', axes=self.axes, preprocessor = self.preprocessor)




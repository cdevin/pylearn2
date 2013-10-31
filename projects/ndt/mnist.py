from pylearn2.datasets.mnist import MNIST


class IndexedMNIST(MNIST):
    def __init__(self, which_set, indexes, center = False, shuffle = False,
            one_hot = False, binarize = False, start = None,
            stop = None, axes=['b', 0, 1, 'c'],
            preprocessor = None,
            fit_preprocessor = False,
            fit_test_preprocessor = False):

        super(IndexedMNIST, self).__init__(which_set = which_set,
                                        center = center,
                                        shuffle = shuffle,
                                        one_hot = one_hot,
                                        binarize = binarize,
                                        start = start,
                                        stop = stop,
                                        axes = axes,
                                        preprocessor = preprocessor,
                                        fit_preprocessor = fit_preprocessor,
                                        fit_test_preprocessor = fit_test_preprocessor)

        if self.X is not None:
            self.X = self.X[indexes]
            self.y = self.y[indexes]

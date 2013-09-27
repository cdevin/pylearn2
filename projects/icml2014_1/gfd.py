import os
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far GFD is "
            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace

class GFD(dense_design_matrix.DenseDesignMatrixPyTables):

    nbTags = 166
    imageShape = (48, 48, 1)
    imageSize = imageShape[0] * imageShape[1] * imageShape[2]

    mapper = {'train': 0, 'valid': 1, 'test': 2}

    data_path = '${PYLEARN2_DATA_PATH}/GFD/pylearn2/'

    def __init__(self, which_set, path = None, axes = ('b', 0, 1, 'c'),
                 preprocessor = None):

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/GFD/pylearn2/h5/'

        # load data
        path = preprocess(path)
        file_n = "{}{}.h5".format(path, which_set)
        self.h5file = tables.openFile(file_n, mode = "r")
        data = self.h5file.getNode('/', "Data")

        view_converter = dense_design_matrix.DefaultViewConverter(self.imageShape,
                                                                        axes)
        super(GFD, self).__init__(X = data.X, y = data.y,
                                    view_converter = view_converter)

        if preprocessor:
            if which_set in ['train', 'train_all']:
                can_fit = True
            preprocessor.apply(self, can_fit)

        self.h5file.flush()


    def get_test_set(self):
        return GFD(which_set = 'test', path = self.path,
                   axes = self.axes, preprocessor = self.preprocessor)


class GFD2(dense_design_matrix.DenseDesignMatrixPyTables):
    nbTags = 166
    imageShape = (48, 48, 1)
    imageSize = imageShape[0] * imageShape[1] * imageShape[2]

    mapper = {'train': 0, 'valid': 1, 'test': 2}

    data_path = '${PYLEARN2_DATA_PATH}/GFD/pylearn2/'


    def __init__(self, which_set, path = None, axes = ('b', 0, 1, 'c'),
                 preprocessor = None):

        assert which_set in self.mapper.keys()

        self.__dict__.update(locals())
        del self.self

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/GFD/pylearn2/h5/'

        # load data
        path = preprocess(path)
        file_n = "{}{}.h5".format(path, which_set)
        self.h5file = tables.openFile(file_n, mode = "r")
        data = self.h5file.getNode('/', "Data")

        view_converter = dense_design_matrix.DefaultViewConverter(self.imageShape,
                                                                        axes)
        super(GFD2, self).__init__(X = data.X, y = data.y,
                                    view_converter = view_converter)

        self.second_y = data.z

        specs, source = self.data_specs
        specs = [specs.components[0]] + [VectorSpace(dim=166), VectorSpace(dim=8)]
        source = list(source) + ['second_targets']
        self.data_specs = (CompositeSpace(specs), tuple(source))

    def get_data(self):
        return self.X, self.y, self.second_y



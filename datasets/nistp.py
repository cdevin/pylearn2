import os
import re
import glob
import gzip
import carray
import warnings
try:
    import tables
except ImportError:
    warnings.warn("Couldn't import tables, so far SVHN is "
            "only supported with PyTables")
import numpy
from theano import config
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load
from pylearn2.utils.string_utils import preprocess


class NISTP(dense_design_matrix.DenseDesignMatrix):


    mapper = {'train': 0, 'valid' : 1, 'test': 2}


    def __init__(self, which_set, path = None, center = False, scale = False,
            start = None, stop = None, axes = ('b', 0, 1, 'c'), one_hot = True,
            preprocessor = None):


        assert which_set in self.mapper.keys()
        if scale:
            warnings.warn("Use scale once when creating carray files for the first time")

        if path is None:
            path = '${PYLEARN2_DATA_PATH}/ift6266h10/data/'
            mode = 'r'
        else:
            mode = 'a'
        # load data
        path = preprocess(path)
        try:
            data_x, data_y = self.load_data(which_set, path)
            assert data_x is not None
            assert data_y is not None
            if one_hot and data_y.ndim == 1:
                raise ValueError("Data is being read from exsiting data that"
                                "is not saved in one_hot format")
            elif not one_hot and data_y.ndim == 64:
                raise ValueError("Data is being read from exsiting data that"
                                "is saved in one_hot format")
        except:
            data_x, data_y = self.make_data(which_set, path, one_hot, scale)


        view_converter = dense_design_matrix.DefaultViewConverter((32, 32, 1),
                                                                        axes)
        super(NISTP, self).__init__(X = data_x, y = data_y,
                                    view_converter = view_converter)

        if preprocessor:
            if which_set == 'train':
                can_fit = True
            preprocessor.apply(self, can_fit)


    @staticmethod
    def load_data(which_set, path):

        data_x = carray.open("{}{}/x/".format(path, which_set))
        data_y = carray.open("{}{}/y/".format(path, which_set))
        return data_x, data_y

    @staticmethod
    def make_data(which_set, path, one_hot, scale):


        def load_data(fname, label = False, one_hot = True, scale = False):
            if fname[-2:] == 'gz':
                f = gzip.open(fname, 'r')
            else:
                f = open(fname, 'r')
            if label:
                data = numpy.fromstring(f.read()[20:], dtype = 'int32')
                if one_hot:
                    one_hot = numpy.zeros((data.shape[0], 62), dtype = 'float32')
                    for i in xrange(data.shape[0]):
                        one_hot[i, data[i]] = 1.
                    data = one_hot
            else:
                data = numpy.fromstring(f.read()[20:], dtype = 'uint8').reshape((-1, 1024)).astype('float32')
                if scale:
                    data /= 255.
            f.close()
            return data


        def atoi(text):
            return int(text) if text.isdigit() else text

        def natural_keys(text):
            """
            credit: http://stackoverflow.com/questions/5967500/how-to-correctly-sort-string-with-number-inside
            """
            return [ atoi(c) for c in re.split('(\d+)', text) ]


        def file_list(which_set, data_path):
            if which_set == 'valid':
                return ["{}PNIST07_valid_data.ft".format(data_path)], ["{}PNIST07_valid_labels.ft".format(data_path)]
            elif which_set == 'test':
                return ["{}PNIST07_test_data.ft".format(data_path)], ["{}PNIST07_test_labels.ft".format(data_path)]
            elif which_set == 'train':
                data = glob.glob("{}PNIST07_train*_data.ft.gz".format(data_path))
                labels = glob.glob("{}PNIST07_train*_labels.ft".format(data_path))
                assert len(data) > 0
                assert len(labels) > 0
                # TODO fix the sort, now is: 0,10
                data.sort(key = natural_keys)
                labels.sort(key = natural_keys)
                return data, labels
            else:
                raise ValueError("Unknow set: {}".format(which_set))

        assert which_set in NISTP.mapper.keys()

        # get list of files
        data_path = preprocess('${PYLEARN2_DATA_PATH}/ift6266h10/data/')
        data_files, label_files = file_list(which_set, data_path)

        # load them in carray
        i = 0.
        for data_file, label_file in zip(data_files, label_files):
            print "loading: {} and {}".format(data_file, label_file)
            if i == 0:
                try:
                    os.mkdir("{}{}".format(path, which_set))
                except:
                    pass
                data_x = carray.carray(load_data(data_file,
                                                    label = False,
                                                    scale = scale),
                        rootdir = "{}{}/x/".format(path, which_set),
                        mode = 'w')
                data_y = carray.carray(load_data(label_file,
                                                label = True,
                                                one_hot = one_hot),
                        rootdir = "{}{}/y/".format(path, which_set),
                        mode = 'w')
                data_x.flush()
                data_y.flush()
            else:
                data_x.append(load_data(data_file,
                                        label = False,
                                        scale = scale))
                data_y.append(load_data(label_file,
                                        label = True,
                                        one_hot = one_hot))
                data_x.flush()
                data_y.flush()
            i += 1

        return data_x, data_y


if __name__ == "__main__":
    data = NISTP('train', scale = True, one_hot = True)
    import ipdb
    ipdb.set_trace()

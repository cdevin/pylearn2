import glob
import numpy
import theano
from pylearn2.utils import serial

def shared_dataset(data_x, data_y, borrow = True, cast_int = True):

    shared_x = theano.shared(numpy.asarray(data_x,
                           dtype=theano.config.floatX),
                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                           dtype=theano.config.floatX),
                 borrow=borrow)

    if cast_int:
        shared_y = theano.tensor.cast(shared_y, 'int32')
    return shared_x, shared_y


class DatasetIterator(object):

    def __init__(self, data_path, which):
        self.files_x = glob.glob(data_path + "*train_x.npy".format(which))
        self.files_y = glob.glob(data_path + "*train_y.npy".format(which))
        self.current_index = 0
        self.num_files = len(self.files_x)

    def init_shared(self):
        data_x = serial.load(self.files_x[0])
        data_y = serial.load(self.files_y[0])
        self.x, self.y = shared_dataset(data_x, data_y, cast_int = False)
        self.current_size = data_x.shape[0]
        del data_x, data_y
        return self.x, self.y

    def __iter__(self):
        return self

    def next(self):
        if self.current_index < self.num_files:
            data_x = serial.load(self.files_x[self.current_index])
            data_y = serial.load(self.files_y[self.current_index])
            self.x.set_value(data_x, borrow = True)
            self.y.set_value(data_y, borrow = True)
            self.current_index += 1
            self.current_size = data_x.shape[0]
            del data_x, data_y
            return self.current_size
        else:
            self.current_index = 0
            raise StopIteration




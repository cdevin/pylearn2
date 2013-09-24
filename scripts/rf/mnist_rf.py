from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial

class MNIST(DenseDesignMatrix):

    def __init__(self, which_set = 'train', start=0, stop=-1):
        x=serial.load("data/{}_x.npy".format(which_set))
        y=serial.load("data/{}_y.npy".format(which_set))
        x = x[start:stop]
        y = y[start:stop]
        super(MNIST, self).__init__(X = x, y = y)


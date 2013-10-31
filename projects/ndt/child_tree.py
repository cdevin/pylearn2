import numpy as np
import copy
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST


DATA_PATH = "results/maxout/"

def splitter(model_path):

    model = serial.load(model_path)

    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    y = T.gt(y, 0.5)
    return function([X], y)


def branch_data(brancher, data, batch_size = 100):

    res = []
    for i in xrange(data.shape[0] / batch_size):
        res.append(brancher(data[i * batch_size : (i+1) * batch_size, :]))
    rem = np.mod(data.shape[0], batch_size)
    if rem != 0:
        res.append(brancher(data[data.shape[0] - rem:, :]))


    res = np.concatenate(res)
    pos = np.nonzero(res)
    neg = np.where(res == 0)
    return pos[0], neg[0]



def get_data(which_set, start = None, stop = None):
    return MNIST(which_set, start=start, stop=stop, one_hot=True)



def save_ds(ds, index, which_set):
    serial.save("{}{}_{}.pkl".format(DATA_PATH, which_set, index), ds)

def tree(data_path, data, which_set, index = 1):


    #while index < 100:
    model = "{}{}.pkl".format(data_path, index)
    sp = splitter(model)
    right, left = branch_data(sp, data.X)

    if index == 1:
        ds = copy.deepcopy(data)
        ds.X = ds.X[right]
        ds.y = ds.y[right]
        tree(data_path, ds, which_set, 3)

        ds = copy.deepcopy(data)
        ds.X = ds.X[left]
        ds.y = ds.y[left]
        tree(data_path, ds, which_set, 2)


    else:
        ds = copy.deepcopy(data)
        ds.X = ds.X[right]
        ds.y = ds.y[right]
        save_ds(ds, index * 2 + 1, which_set)

        ds = copy.deepcopy(data)
        ds.X = ds.X[left]
        ds.y = ds.y[left]
        save_ds(ds, index * 2, which_set)


if __name__ == "__main__":
        ds = get_data('train', 0, 50000)
        tree(DATA_PATH, ds, 'train')


        ds = get_data('train', 50000, 60000)
        tree(DATA_PATH, ds, 'valid')

        ds = get_data('test')
        tree(DATA_PATH, ds, 'test')

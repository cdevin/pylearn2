import numpy as np
import copy
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils.string_utils import preprocess

#DATA_PATH = "results/maxout/"
DATA_PATH = "/RQexec/mirzameh/results/tree/cifar10/maxout0/"

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

def branch_datac01b(brancher, data, batch_size = 100):

    res = []
    print data.shape
    data = data.astype('float32')
    for i in xrange(data.shape[3] / batch_size):
        res.append(brancher(data[:,:,:,i * batch_size : (i+1) * batch_size]))
    rem = np.mod(data.shape[3], batch_size)
    if rem != 0:
        res.append(brancher(data[:,:,:,data.shape[3] - rem:]))


    res = np.concatenate(res)
    pos = np.nonzero(res)
    neg = np.where(res == 0)
    return pos[0], neg[0], res

def get_data(which_set, start = None, stop = None):
    return MNIST(which_set, start=start, stop=stop, one_hot=True)

def get_cifar(which_set, start = None, stop = None):
    loc = "${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/"
    loc = preprocess(loc)
    return ZCA_Dataset(preprocessed_dataset=serial.load("{}{}.pkl".format(loc, which_set)),
                preprocessor=serial.load("{}/preprocessor.pkl".format(loc)),
                start=start,
                stop=stop,
                axes=['c', 0, 1, 'b'])


def save_ds(ds, index, which_set):
    serial.save("{}{}_{}.pkl".format(DATA_PATH, which_set, index), ds)

def tree(data_path, data, which_set, index = 1, topo = False):


    #while index < 100:
    model = "{}{}.pkl".format(data_path, index)
    sp = splitter(model)
    if topo:
        right, left, res = branch_datac01b(sp, data.get_topological_view())
    else:
        right, left = branch_data(sp, data.X)

    print right.shape, left.shape, which_set
    r_ = (res * data.y).sum(axis=0)
    l_ = ((np.negative(res) + 1) * data.y).sum(axis=0)
    print np.argmax(np.vstack((r_, l_)), 0)
    print r_
    print l_
    return

    if index == 1:
        ds = copy.deepcopy(data)
        ds.X = ds.X[right]
        ds.y = ds.y[right]
        #tree(data_path, ds, which_set, 3)
        save_ds(ds, index * 2 + 1, which_set)

        ds = copy.deepcopy(data)
        ds.X = ds.X[left]
        ds.y = ds.y[left]
        #tree(data_path, ds, which_set, 2)
        save_ds(ds, index * 2, which_set)


    else:
        ds = copy.deepcopy(data)
        ds.X = ds.X[right]
        ds.y = ds.y[right]
        save_ds(ds, index * 2 + 1, which_set)

        ds = copy.deepcopy(data)
        ds.X = ds.X[left]
        ds.y = ds.y[left]
        save_ds(ds, index * 2, which_set)



def do_mnist():

    ds = get_data('train', 0, 50000)
    tree(DATA_PATH, ds, 'train')

    ds = get_data('train', 50000, 60000)
    tree(DATA_PATH, ds, 'valid')

    ds = get_data('test')
    tree(DATA_PATH, ds, 'test')

def do_cifar():

    ds = get_cifar('train', 0, 50000)
    tree(DATA_PATH, ds, 'train', topo=True)


    ds = get_cifar('train', 40000, 50000)
    tree(DATA_PATH, ds, 'valid', topo=True)

    ds = get_cifar('test')
    tree(DATA_PATH, ds, 'test', topo=True)




if __name__ == "__main__":
    do_cifar()

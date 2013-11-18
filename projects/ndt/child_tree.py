import numpy as np
import copy
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils.string_utils import preprocess
from noisylearn.projects.ndt.zca_dataset import ZCA_Dataset_BIN

#DATA_PATH = "results/maxout/"
#DATA_PATH = "/RQexec/mirzameh/results/tree/cifar10/maxout0/"
#DATA_PATH = "/RQexec/mirzameh/results/tree/cifar10_bin/"
DATA_PATH = "/RQexec/mirzameh/results/tree/cifar10/"

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

def branch_datac01b_bin(brancher, data, batch_size = 100):

    res = []
    print data.shape
    data = data.astype('float32')
    for i in xrange(data.shape[3] / batch_size):
        res.append(brancher(data[:,:,:,i * batch_size : (i+1) * batch_size]))
    rem = np.mod(data.shape[3], batch_size)
    if rem != 0:
        res.append(brancher(data[:,:,:,data.shape[3] - rem:]))


    res = np.concatenate(res)
    pos = np.nonzero(np.argmax(res,1))
    neg = np.nonzero(np.argmin(res,1))
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

def get_cifar_bin(which_set, start = None, stop = None):
    loc = "${PYLEARN2_DATA_PATH}/cifar10/pylearn2_gcn_whitened/"
    loc = preprocess(loc)
    return ZCA_Dataset_BIN(preprocessed_dataset=serial.load("{}{}.pkl".format(loc, which_set)),
                preprocessor=serial.load("{}/preprocessor.pkl".format(loc)),
                labels= [1,1,0,0,0,0,0,0,1,1],
                start=start,
                stop=stop,
                axes=['c', 0, 1, 'b'])

def save_ds(ds, index, which_set):
    serial.save("{}{}_{}.pkl".format(DATA_PATH, which_set, index), ds)

def tree(data_path, data, which_set, index = 1, dstype = 'vector'):


    #while index < 100:
    model = "{}{}.pkl".format(data_path, index)
    sp = splitter(model)
    if dstype == 'topo':
        right, left, res = branch_datac01b(sp, data.get_topological_view())
    elif dstype == 'vector':
        right, left = branch_data(sp, data.X)
    elif dstype == 'topobin':
        right, left, res = branch_datac01b_bin(sp, data.get_topological_view())
    else:
        raise NameError("Bad dstype: {}".format(dstype))

    print right.shape, left.shape, which_set
    #r_ = (res * data.y).sum(axis=0)
    #l_ = ((np.negative(res) + 1) * data.y).sum(axis=0)
    #print np.argmax(np.vstack((r_, l_)), 0)
    #print r_
    #print l_
    #import ipdb
    #ipdb.set_trace()
    #return

    print "Index is: {}".format(index)
    if index == 1:
        ds = copy.deepcopy(data)
        ds.X = ds.X[right]
        ds.y = ds.y[right]
        save_ds(ds, index * 2 + 1, which_set)
        tree(data_path, ds, which_set, 3, dstype = dstype)

        ds = copy.deepcopy(data)
        ds.X = ds.X[left]
        ds.y = ds.y[left]
        save_ds(ds, index * 2, which_set)
        tree(data_path, ds, which_set, 2, dstype = dstype)


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
    tree(DATA_PATH, ds, 'train', dstype='topo')

    ds = get_cifar('train', 40000, 50000)
    tree(DATA_PATH, ds, 'valid', dstype='topo')

    ds = get_cifar('test')
    tree(DATA_PATH, ds, 'test',  dstype='topo')

def do_cifar_bin():

    ds = get_cifar('train', 0, 50000)
    tree(DATA_PATH, ds, 'train', dstype='topobin')

    ds = get_cifar('train', 40000, 50000)
    tree(DATA_PATH, ds, 'valid', dstype='topobin')

    ds = get_cifar('test')
    tree(DATA_PATH, ds, 'test', dstype='topobin')


if __name__ == "__main__":
    do_cifar()
    #do_cifar_bin()

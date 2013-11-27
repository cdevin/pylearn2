import os
import numpy as np
import copy
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.zca_dataset import ZCA_Dataset
from pylearn2.utils.string_utils import preprocess
from noisylearn.projects.ndt.zca_dataset import ZCA_Dataset_BIN
from noisylearn.projects.ndt.zca_dataset import Indexed_ZCA_Dataset

DATA_PATH = preprocess("${PYLEARN2_EXP_RESULTS}/tree/cifar10/")

def splitter(model_path):

    model = serial.load(model_path)

    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    y = T.gt(y, 0.5)
    return function([X], y)


def model_output(model_path):

    model = serial.load(model_path)
    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)

    return function([X], y)

def get_confidence_c01b(brancher, data, batch_size = 100):

    res = []
    print data.shape
    data = data.astype('float32')
    for i in xrange(data.shape[3] / batch_size):
        res.append(brancher(data[:,:,:,i * batch_size : (i+1) * batch_size]))
    rem = np.mod(data.shape[3], batch_size)
    if rem != 0:
        res.append(brancher(data[:,:,:,data.shape[3] - rem:]))

    res = np.concatenate(res)
    return  res


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

def save_indexes(path, indexes, which_set, nodeindex):
    print "{}{}_{}.npy".format(path, which_set, nodeindex)
    serial.save("{}{}_{}.npy".format(path, which_set, nodeindex), indexes)

def tree(data_path, data, which_set, index = 1, dstype = 'vector'):

    print "Index is: {}".format(index)
    # TODO temp hack remove me
    if index == 1:
        dstype = 'topobin'
    else:
        dstype = 'topo'

    model = "{}{}.pkl".format(data_path, index)
    if not os.path.isfile(model):
        print "No -{}- node found".format(index)
        return

    sp = splitter(model)
    #sp = model_output(model)
    if hasattr(data, 'indexes'):
        x = data.X[data.indexes]
        y = data.y[data.indexes]
    else:
        x = data.X
        y = data.y
    if dstype == 'topo':
        right, left, res = branch_datac01b(sp, data.get_topological_view(x))
        #res = get_confidence_c01b(sp, data.get_topological_view(x))
        print 'hiii'
    elif dstype == 'vector':
        right, left = branch_data(sp, x)
    elif dstype == 'topobin':
        right, left, res = branch_datac01b_bin(sp, data.get_topological_view(x))
    else:
        raise NameError("Bad dstype: {}".format(dstype))


    ### Temp for saving results
    np.save("res_out.npy", res)
    return
    ####


    print right.shape, left.shape, which_set
    if dstype == 'topobin':
        r_ = (np.argmax(res,1).reshape((res.shape[0],1)) * y).sum(axis=0)
        l_ = (np.argmin(res,1).reshape((res.shape[0],1)) * y).sum(axis=0)
    else:
        r_ = (res * y).sum(axis=0)
        l_ = ((np.negative(res) + 1) * y).sum(axis=0)

    print np.argmax(np.vstack((r_, l_)), 0)
    print r_
    print l_


    if index == 1:
        save_indexes(data_path, left, which_set, index * 2)
        save_indexes(data_path, right, which_set, index * 2 + 1)

        data.indexes = left
        tree(data_path, data, which_set, 2, dstype = dstype)
        data.indexes = right
        tree(data_path, data, which_set, 3, dstype = dstype)

    else:
        left = orig_index(data, left)
        right = orig_index(data, right)

        save_indexes(data_path, left, which_set, index * 2)
        save_indexes(data_path, right, which_set, index * 2 + 1)

        data.indexes = left
        tree(data_path, data, which_set, index * 2, dstype = dstype)
        data.indexes = right
        tree(data_path, data, which_set, index * 2 + 1, dstype = dstype)

def orig_index(dataset, indexes):

    if not hasattr(dataset, 'indexes'):
        raise TypeError("Dataset should be indexed")
    return dataset.indexes[indexes]

def do_mnist():

    ds = get_data('train', 0, 50000)
    tree(DATA_PATH, ds, 'train')

    ds = get_data('train', 50000, 60000)
    tree(DATA_PATH, ds, 'valid')

    ds = get_data('test')
    tree(DATA_PATH, ds, 'test')

def do_cifar():

    #ds = get_cifar('train', 0, 40000)
    #tree(DATA_PATH, ds, 'train', dstype='topo')
#
    #ds = get_cifar('train', 40000, 50000)
    #tree(DATA_PATH, ds, 'valid', dstype='topo')

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

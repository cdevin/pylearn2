""" This module save represntaions of models in different format
"""

import argparse, os, fnmatch
import numpy
import theano
from scipy.io import savemat
from theano import tensor
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
from pylearn2.datasets.cifar10 import CIFAR10
from utils.datasets.cifar10_bw import CIFAR10_BW
from sklearn.preprocessing import Scaler

def load_data(data_path):

    return serial.load(data_path)

def save(data, path):

    for ind, item in enumerate(data):
        path + 'l1'
        numpy.save(path + 'l{}.npy'.format(ind), item)

def convert(data, model_f):

    print "Transforming data..."
    x = tensor.matrix()
    model = serial.load(model_f)
    l1 = model.hidden_layers[0].test_encode(x)
    l2 = model.hidden_layers[1].test_encode(l1)
    fn = theano.function([x], [l1, l2])
    return fn(data.X)


def report(data):

    for ind, item in enumerate(data):
        zeros = numpy.mean([numpy.count_nonzero(row == 0) for row in item])
        print 'Layer {} mean numb of zeros is {}/{}'.format(ind, zeros, item.shape[1])
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'save features of model')
    parser.add_argument('-d', '--data', help = "path to data file", required=True)
    parser.add_argument('-m', '--model', help = "path to model file", required=True)
    parser.add_argument('-s', '--save', help = "save path")
    args = parser.parse_args()


    # load datasets
    data = load_data(args.data)
    feats = convert(data, args.model)
    report(feats)
    save(feats, args.save)

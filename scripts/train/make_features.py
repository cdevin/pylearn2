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

def norm(X):
    s = X.std(0)
    m = X.mean(0)
    s = s + 0.0001 * (s==0)
    return (X-m)/s

def load_data(dataset):

    print "Loading data..."
    if dataset == 'mnist':
        train_set = MNIST('train')
        test_set = MNIST('test')
    elif dataset == 'cifar10_bw':
        train_set = CIFAR10_BW('train')
        test_set = CIFAR10_BW('test')
    elif dataset == 'cifar10':
        train_set = CIFAR10('train')
        test_set = CIFAR10('test')
    else:
        raise NameError('Unknown dataset: {}').format(dataset)

    return train_set, test_set

def save(train, test, path, name, file_format):

    if file_format == 'mat':
        if name == 'labels':
            data = {'train_y' : train, 'test_y' : test}
            savemat(path + 'labels', data)
        else:
            data = {'train_x': train, 'test_x' : test}
            savemat(path + name, data)
    else:
        raise NameError('Unknown format: {}'.format(file_format))

def convert(train_x, test_x, model_f, dataset, scale, normalize):

    if model_f != None:
        print "Transforming data..."
        x = tensor.matrix()
        rep = serial.load(model_f)
        rep.fn = theano.function([x], rep.test_encode(x))

        train_feat = rep.perform(train_x)
        test_feat = rep.perform(test_x)

    if scale:
        print "Scaling data..."
        scaler = Scaler()
        scaler.fit(train_feat)
        train_feat = scaler.transform(train_feat)
        test_feat = scaler.transform(test_feat)

    if normalize:
        print "Normalizing..."
        train_feat = numpy.vstack([norm(x) for x in train_feat])
        test_feat = numpy.vstack([norm(x) for x in test_feat])

    return train_feat, test_feat




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'save features of model')
    parser.add_argument('-p', '--path', help = "path to model files", required=True)
    parser.add_argument('--pattern', help = "pattern of files", default = "*.pkl")
    parser.add_argument('-s', '--scale', action = "store_true", default = False, help = "scale data")
    parser.add_argument('-n', '--norm', action = "store_true", default = False, help = "normalize data")
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10_bw', 'cifar10'], required = True)
    parser.add_argument('-f', '--format', choices = ['mat'], default = 'mat')
    args = parser.parse_args()

    # find files
    matches = []
    for root, dirnames, filenames in os.walk(args.path):
        for filename in fnmatch.filter(filenames, args.pattern):
            matches.append(os.path.join(root, filename))


    # load datasets
    train_set, test_set = load_data(args.dataset)

    # save features
    for item in matches:
        train_feat, test_feat = convert(train_set.X, test_set.X, item, args.dataset, args.scale, args.norm)
        save_path = item.rstrip(item.split('/')[-1])
        name = item.split('/')[-1].rstrip('.pkl')
        save(train_feat, test_feat, save_path, name, args.format)


    # save labels
    save(train_set.y, test_set.y, args.path, 'labels', args.format)

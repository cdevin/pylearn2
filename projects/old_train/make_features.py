""" This module save represntaions of models in different format
"""

import argparse, os, fnmatch
import numpy
import theano
from scipy.io import savemat
from theano import tensor
from pylearn2.utils import serial
from sklearn.preprocessing import Scaler

def norm(X):
    s = X.std(0)
    m = X.mean(0)
    s = s + 0.0001 * (s==0)
    return (X-m)/s

def load_data(dataset):

    print "Loading data..."
    return serial.load(dataset)


def save(data, path, file_format):

    if file_format == 'mat':
        if name == 'labels':
            data = {'train_y' : train, 'test_y' : test}
            savemat(path + 'labels', data)
        else:
            data = {'train_x': train, 'test_x' : test}
            savemat(path + name, data)

    elif file_format == 'npy':
        numpy.save(path, data)
    else:
        raise NameError('Unknown format: {}'.format(file_format))

def convert(data, batch_size, model_f, dataset):

    print "Transforming data..."
    x = tensor.matrix()
    model = serial.load(model_f)
    res = model.conv_encode(x)
    for layer in model.mlp.hiddens.layers:
        res = layer.test_encode(res)
    fn = theano.function([x], res)

    feat = []
    n_batches = data.shape[0] / batch_size
    for i in xrange(n_batches):
        feat.append(fn(data[i*batch_size:(i+1) * batch_size]))
    if numpy.mod(data.shape[0], batch_size) != 0:
        rem = numpy.concatenate([data[n_batches * batch_size :],
            numpy.random.random((batch_size - numpy.mod(data.shape[0], batch_size), data.shape[1])).astype('float32')])
        feat.append(fn(rem))

    feat = numpy.concatenate(feat)[:data.shape[0]]
    return feat



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'save features of model')
    parser.add_argument('-p', '--path', help = "path to model files", required=True)
    parser.add_argument('-s', '--save_path', help = "path to save", required=True)
    parser.add_argument('-d', '--dataset',  required = True)
    parser.add_argument('-f', '--format', choices = ['mat', 'npy'], default = 'npy')
    args = parser.parse_args()

    # load datasets
    data = load_data(args.dataset)

    # save features
    feat = convert(data.X, 20, args.path, args.dataset)
    save(feat, args.save_path, args.format)


    # save labels
    save_path = args.save_path.rstrip('.npy') + '_y.npy'
    save(data.y, save_path, args.format)

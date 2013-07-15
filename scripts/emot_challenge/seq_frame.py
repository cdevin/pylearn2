"""
Get sequence classification reuslts based on single based classifiers
"""

import numpy
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.datasets import preprocessing
import theano.tensor as T
from theano import function
import sys
#from emotiw.common.datasets.faces.afew2_facetubes import AFEW2FaceTubes
from afew2 import AFEW2FaceTubes
import ipdb

def get_classifier(model):

    X = model.get_input_space().make_batch_theano()
    X.name = 'X'

    y = model.fprop(X)
    y.name = 'y'

    return function([X],y)

def accuracy(predictions, target):

    mx = numpy.argmax(predictions, axis=1)
    mask = numpy.zeros(predictions.shape)

    for i in range(mask.shape[0]):
        mask[i, mx[i]] = 1.
    predictions = mask * predictions
    y_hat = numpy.argmax(predictions.sum(axis=0))

    #max = predictions.max(axis=0)
    #hapoo = numpy.log(numpy.exp(predictions-max[numpy.newaxis,:]).sum(axis=0))
    #y_hat = numpy.argmax(hapoo)
    print y_hat, target
    return int(y_hat != target)


def get_stats(predictions):

    stats=[]

    #
    stats.append(predictions.mean(axis=0))

    #
    max = predictions.max(axis=0)
    stats.append(numpy.log(numpy.exp(predictions-max[numpy.newaxis,:]).sum(axis=0)))

    #
    mx = numpy.argmax(predictions, axis=1)
    mask = numpy.zeros(predictions.shape)
    for i in range(mask.shape[0]):
        mask[i, mx[i]] = 1.
    stats.append((mask * predictions).sum(axis=0))

    return numpy.concatenate(stats)

def get_data(which):
    data = AFEW2FaceTubes(which_set='valid', one_hot=True, preproc=['smooth'], size=[48,48], source=which, prep='_prep')

    # organize axis
    data_axes = data.data_specs[0].components[0].axes
    if 't' in data_axes:
        data_axes = [axis for axis in data_axes if axis not in ['t']]

    targets = numpy.argmax(data.targets, axis=1)
    return data.features, targets, data_axes


if __name__ == "__main__":

    _, model_path = sys.argv
    model = serial.load(model_path)
    batch_size = model.batch_size
    classifier = get_classifier(model)

    model_axes = model.input_space.axes
    if model_axes != ('c', 0, 1, 'b'):
        raise ValueError("Model axis is not supoorted")

    features, targets, data_axes = get_data('samira')

    misclass = []
    stats = []
    for feature, target in zip(features, targets):
        #feature = feature / 255.
        feature = feature.astype('float32')
        if data_axes != model_axes:
            feature = feature.transpose(*[data_axes.index(axis) for axis in model_axes])

        num_samples = feature.shape[3]
        predictions = []
        for i in range(num_samples / batch_size):
            # TODO FIX ME, after grayscale
            predictions.append(classifier(feature[0,:,:,i*batch_size:(i+1)*batch_size][numpy.newaxis,:,:,:]))

        # for modulo we padd with garbage
        if batch_size > num_samples:
            modulo = batch_size - num_samples
        else:
            modulo = num_samples % batch_size

        if modulo != 0:
            # TODO FIX ME, after grayscale
            shape = [1, feature.shape[1], feature.shape[2], modulo]
            padding = numpy.ones((shape)).astype('float32')
            # TODO FIX ME, after grayscale
            feature = numpy.concatenate((feature[0,:,:,(num_samples/batch_size) * batch_size:][numpy.newaxis,:,:,:], padding), axis = 3)
            predictions.append(classifier(feature)[:batch_size - modulo])
        predictions = numpy.concatenate(predictions, axis=0)
        misclass.append(accuracy(predictions, target))
        #stats.append(get_stats(predictions))

    error = numpy.sum(misclass) / float(len(features))
    print error, 1-error

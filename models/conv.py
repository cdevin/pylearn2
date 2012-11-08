"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import load_data as mnist_load_data
from EmotionDetector.util.funcs import load_data as tfd_load_data
from noisy_encoder.utils.corruptions import BinomialCorruptorScaled
from pylearn2.corruption import GaussianCorruptor

import jobman

class HiddenLayer(object):
    def __init__(self,
                rng,
                input_clean,
                input_corrupted,
                n_in,
                n_out,
                W=None,
                b=None,
                activation=T.tanh):

        self.input_clean = input_clean
        self.input_corrupted = input_corrupted

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output_clean = T.dot(input_clean, self.W) + self.b
        lin_output_corrupted = T.dot(input_corrupted, self.W) + self.b
        self.output_clean = (lin_output_clean if activation is None
                       else activation(lin_output_clean))
        self.output_corrupted = (lin_output_corrupted if activation is None
                       else activation(lin_output_corrupted))
        # parameters of the model
        self.params = [self.W, self.b]

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input_clean, input_corrupted, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.p_y_given_x_clean = T.nnet.softmax(T.dot(input_clean, self.W) + self.b)
        self.p_y_given_x_corrupted = T.nnet.softmax(T.dot(input_corrupted, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x_clean, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x_corrupted)[T.arange(y.shape[0]), y])

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input_clean, input_corrupted, filter_shape, image_shape, poolsize, activation = T.tanh):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        assert image_shape[1] == filter_shape[1]
        self.input_clean = input_clean
        self.input_corrupted = input_corrupted

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out_clean = conv.conv2d(input=input_clean, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        conv_out_corrupted = conv.conv2d(input=input_corrupted, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out_clean = downsample.max_pool_2d(input=conv_out_clean,
                                            ds=poolsize, ignore_border=True)
        pooled_out_corrupted = downsample.max_pool_2d(input=conv_out_corrupted,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output_clean = activation(pooled_out_clean + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.output_corrupted = activation(pooled_out_corrupted + self.b.dimshuffle('x', 0, 'x', 'x'))
        # store parameters of this layer
        self.params = [self.W, self.b]

def load_data(dataset, fold = 0, center = False, scale = True):
    if dataset == 'tfd':
        return tfd_load_data('train', fold, center, scale),\
            tfd_load_data('valid', fold, center, scale),\
            tfd_load_data('test', fold, center, scale)
    else:
        return mnist_load_data('/RQexec/mirzameh/data/mnist/mnist.pkl.gz')

def eval_activation(activation):
    if activation == 'tanh':
        return T.tanh
    elif activation == 'sigmoid':
        return T.nnet.sigmoid
    elif activation == 'rectifier':
        def rectifier(X):
            return X * (X > 0.0)
        return rectifier

class Conv(object):
    def __init__(self,
                    rng,
                    image_shapes,
                    nkenrs,
                    filter_shapes,
                    poolsizes,
                    binomial_corruption_levels,
                    gaussian_corruption_levels,
                    nhid,
                    nout,
                    activation,
                    batch_size=500):

        activation = eval_activation(activation)

        x = T.matrix('x')   # the data is presented as rasterized images
        y = T.ivector('y')  # the labels are presented as 1D vector of


        print '... building the model'

        n_layers = len(filter_shapes)

        # conv layers
        self.layers = []
        self.params = []
        for i in range(n_layers):
            if i == 0:
                input_clean = x.reshape((batch_size, 1, image_shapes[0][1], image_shapes[0][1]))
                input_corrupted = GaussianCorruptor(stdev = \
                        gaussian_corruption_levels[0])(x.reshape((batch_size, 1,
                                image_shapes[0][0], image_shapes[0][1])))
            else:
                input_clean = self.layers[-1].output_clean
                input_corrupted = GaussianCorruptor(stdev = \
                        gaussian_corruption_levels[i])(self.layers[-1].output_corrupted)
                input_corrupted = BinomialCorruptorScaled(corruption_level = \
                        binomial_corruption_levels[i-1])(input_corrupted)

            layer = LeNetConvPoolLayer(rng, input_clean = input_clean,
                                    input_corrupted = input_corrupted,
                                    image_shape=(batch_size, nkerns[i],
                                        image_shapes[i][0], image_shapes[i][1]),
                                    filter_shape=filter_shapes[i],
                                    poolsize=poolsizes[i],
                                    activation = activation)
            self.layers.append(layer)
            self.params.extend(layer.params)

        # Hidden layer
        input_clean = self.layers[-1].output_clean.flatten(2)
        input_corrupted = GaussianCorruptor(stdev = \
                gaussian_corruption_levels[-2])(self.layers[-1].output_corrupted.flatten(2))
        input_corrupted = BinomialCorruptorScaled(corruption_level = \
                binomial_corruption_levels[-2])(input_corrupted)
        self.hid_layer = HiddenLayer(rng, input_clean=input_clean,
                    input_corrupted = input_corrupted,
                    n_in=nkerns[-1] * numpy.prod(image_shapes[-1]),
                    n_out=nhid, activation= activation)

        # Logistic layer
        input_corrupted = GaussianCorruptor(stdev = \
                gaussian_corruption_levels[-1])(hid_layer.output_corrupted)
        input_corrupted = BinomialCorruptorScaled(corruption_level = \
                binomial_corruption_levels[-1])(input_corrupted)
        self.log_layer = LogisticRegression(input_clean=hid_layer.output_clean,
                input_corrupted = input_corrupted, n_in=nhid, n_out=nout)


    def build_fintune_function(self, datasets, batch_size, enable_momnetum):

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches /= batch_size


        index = T.lscalar()  # index to a [mini]batch
        learning_rate = T.scalar('lr')
        if enable_momnetum is None:
            momentum = None
        else:
            momentum = T.scalar('momentum')
        cost = self.log_layer.negative_log_likelihood(y)
        gparams = T.grad(cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self.params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.params, gparams):
                inc = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc - learning_rate * gparam
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score





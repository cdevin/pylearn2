import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input_clean, input_corrupted, n_in, n_out, irange = 0.01, bias_init = 1.0, rng = 9001):

        if rng == None or type(rng) == type(1):
            rng = numpy.random.RandomState(rng)

        self.W = sharedX(rng.normal(loc = 0.0,
            scale = irange,
            size = (n_in, n_out)),
            name='W', borrow=True )
        self.b = sharedX(
            numpy.ones(n_out) * bias_init,
            name='b',
            borrow=True)

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


class HiddenLayer(object):

    def __init__(self, input_clean, input_corrupted, n_in, n_out, activation, irange = 0.01, bias_init = 1.0, rng = 9001):

        if rng == None or type(rng) == type(1):
            rng = numpy.random.RandomState(rng)

        self.W = sharedX(rng.normal(loc = 0.0,
            scale = irange,
            size = (n_in, n_out)),
            name='W', borrow=True )
        self.b = sharedX(
            numpy.ones(n_out) * bias_init,
            name='b',
            borrow=True)

        lin_output_clean = T.dot(input_clean, self.W) + self.b
        lin_output_corrupted = T.dot(input_corrupted, self.W) + self.b
        self.output_clean = (lin_output_clean if activation is None
                       else activation(lin_output_clean))
        self.output_corrupted = (lin_output_corrupted if activation is None
                       else activation(lin_output_corrupted))
        # parameters of the model
        self.params = [self.W, self.b]


class PickableLambda(object):
    def __call__(self, x):
        return x

def rectifier(X):
    return X * (X > 0.0)

def eval_activation(activation):
    if activation == 'tanh':
        return T.tanh
    elif activation == 'sigmoid':
        return T.nnet.sigmoid
    elif activation == 'rectifier':
        def rectifier(X):
            return X * (X > 0.0)
        return rectifier



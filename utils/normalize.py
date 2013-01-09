import numpy
import theano
from theano import tensor
from theano.tensor.nnet.conv import conv2d
from pylearn2.utils import sharedX




class LocalResponseNormalize(object):

    def __init__(self, batch_size, image_size, n, alpha, beta):
        self.batch_size = batch_size
        self.image_size = image_size
        self.n = n
        self.alpha = alpha
        self.beta = beta

        self.filters =  sharedX(numpy.ones((1, 1, n, n), dtype = theano.config.floatX))


    def _apply(self, x):
        """
        out = x / ((1 + (alpha / n **2) * conv(x ** 2, n) )) ** beta
        """

        base = conv2d(x ** 2, self.filters, filter_shape = (1, 1, self.n, self.n),
                image_shape = (self.batch_size, 1, self.image_size, self.image_size),
                border_mode = 'full')

        new_size = self.image_size + self.n -1
        pad_r = int(numpy.ceil((self.n-1) / 2.))
        pad_l = self.n-1 - pad_r
        pad_l = int(self.image_size + self.n -1 - pad_l)

        base = base[:,:, pad_r:pad_l, pad_r:pad_l]
        base = 1 + (self.alpha / self.n**2) * base
        return x / (base ** self.beta)


    def __call__(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self._apply(inputs)
        return [self._apply(inp) for inp in inputs]


def test_local_response_normalize():

    x = tensor.tensor4('x')
    batch_size = 8
    image_size = 10
    n = 4
    alpha = 0.0001
    beta = 0.75

    norm = LocalResponseNormalize(batch_size, image_size,  n, alpha, beta)(x)
    f = theano.function([x], norm)

    x_ = numpy.random.random((batch_size, 1, image_size, image_size)).astype('float32')
    res = f(x_)
    assert x_.shape == res.shape



if __name__ == "__main__":
    test_local_response_normalize()

import numpy
import theano
from theano import tensor as T
from theano import function
from noisy_encoder.models.convlinear import stochastic_max_pool_c01b, weighted_sum_pool_c01b

def test_stochastic_max_pool():
    theano.config.compute_test_value = 'warn'
    inp = numpy.random.random((3, 32, 32, 10)).astype('float32')
    x = T.tensor4()
    x.tag.test_value = inp
    image_shape = [32, 32]
    pool_shape = [3, 3]
    pool_stride = [1, 1]
    y=stochastic_max_pool_c01b(x, pool_shape, pool_stride, image_shape)
    f = function([x],y)

    print f(inp).shape

def testweighted_max_pool():
    theano.config.compute_test_value = 'warn'
    inp = numpy.random.random((3, 32, 32, 10)).astype('float32')
    x = T.tensor4()
    x.tag.test_value = inp
    image_shape = [32, 32]
    pool_shape = [3, 3]
    pool_stride = [1, 1]
    y=weighted_sum_pool_c01b(x, pool_shape, pool_stride, image_shape)
    f = function([x],y)

    print f(inp).shape

if __name__ == "__main__":
    testweighted_max_pool()
    #test_stochastic_max_pool()

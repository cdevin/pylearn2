"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from collections import OrderedDict
import numpy
import warnings
import time

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor
import theano

from pylearn2.costs.cost import Cost
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import safe_izip
from pylearn2.utils import sharedX


def stochastic_max_pool(bc01, pool_shape, pool_stride, image_shape, rng = None):
    """
    Stochastic max pooling for training as defined in:

    Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    Matthew D. Zeiler, Rob Fergus

    bc01: minibatch in format (batch size, channels, rows, cols),
        IMPORTANT: All values should be poitivie
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    batch = bc01.shape[0]
    channel = bc01.shape[1]

    #if rng is None:
        #rng = RandomStreams(2022)

    if rng is None:
        rng = MRG_RandomStreams(2022)

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(numpy.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    # final result shape
    res_r = int(numpy.floor(last_pool_r/rs)) + 1
    res_c = int(numpy.floor(last_pool_c/cs)) + 1

    for bc01v in get_debug_values(bc01):
        assert not numpy.any(numpy.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    # padding
    padded = tensor.alloc(0.0, batch, channel, required_r, required_c)
    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = tensor.set_subtensor(padded[:,:, 0:r, 0:c], bc01)
    bc01.name = 'zero_padded_' + name

    # unraveling
    window = tensor.alloc(0.0, batch, channel, res_r * res_c, pr, pc)
    window.name = 'unravlled_winodows_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            win_cell = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            win_cell = win_cell.reshape((batch, channel, res_r * res_c))
            window  =  tensor.set_subtensor(window[:,:, :, row_within_pool, col_within_pool], win_cell)

    # find the norm
    norm = window.sum(axis = [3, 4])
    norm = tensor.switch(tensor.eq(norm, 0.0), 1.0, norm)
    norm = window / norm.dimshuffle(0, 1, 2, 'x', 'x')
    # get prob
    prob = rng.multinomial(pvals = norm.reshape((batch * channel * res_r * res_c, pr * pc)))
    # select
    res = (window * prob.reshape((batch, channel, res_r * res_c,  pr, pc))).max(axis=4).max(axis=3)
    res.name = 'pooled_' + name

    return tensor.cast(res.reshape((batch, channel, res_r, res_c)), theano.config.floatX)


def stochastic_max_pool2(bc01, pool_shape, pool_stride, image_shape, rng = None):
    """
    Stochastic max pooling for training as defined in:

    Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    Matthew D. Zeiler, Rob Fergus

    bc01: minibatch in format (batch size, channels, rows, cols),
        IMPORTANT: All values should be poitivie
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    batch = bc01.shape[0]
    channel = bc01.shape[1]

    #if rng is None:
        #rng = RandomStreams(2022)

    if rng is None:
        rng = MRG_RandomStreams(2022)

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(numpy.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    # final result shape
    res_r = int(numpy.floor(last_pool_r/rs)) + 1
    res_c = int(numpy.floor(last_pool_c/cs)) + 1

    for bc01v in get_debug_values(bc01):
        assert not numpy.any(numpy.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    # padding
    padded = tensor.alloc(0.0, batch, channel, required_r, required_c)
    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = tensor.set_subtensor(padded[:,:, 0:r, 0:c], bc01)
    bc01.name = 'zero_padded_' + name

    # unraveling
    window = tensor.alloc(0.0, batch, channel, res_r, res_c, pr, pc)
    window.name = 'unravlled_winodows_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            win_cell = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            window  =  tensor.set_subtensor(window[:,:, :, :, row_within_pool, col_within_pool], win_cell)

    # find the norm
    norm = window.sum(axis = [4, 5])
    norm = tensor.switch(tensor.eq(norm, 0.0), 1.0, norm)
    norm = window / norm.dimshuffle(0, 1, 2, 3, 'x', 'x')
    # get prob
    prob = rng.multinomial(pvals = norm.reshape((batch * channel * res_r * res_c, pr * pc)))
    # select
    res = (window * prob.reshape((batch, channel, res_r, res_c,  pr, pc))).max(axis=5).max(axis=4)
    res.name = 'pooled_' + name

    return tensor.cast(res, theano.config.floatX)



def numpy_stochastic_max_pool(bc01, pool_shape, pool_stride, image_shape):
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    wide_infinity = np.ones((bc01.shape[0], bc01.shape[1], required_r, required_c)) * -np.inf
    #import ipdb
    #ipdb.set_trace()
    wide_infinity[:,:, 0:r, 0:c] = bc01
    bc01 = wide_infinity
    res = np.zeros((bc01.shape[0], bc01.shape[1], np.floor(required_r/rs), np.fllor(required_c / rc)))

    for r in xrange(0, required_r, rs):
        for c in xrange(0, required_c, rc):
            window = bc01[:,:, r:r+pr, c:c+pc]
            # normalize
            window = window / window.sum(axis=3).sum(axis=2)

def numpy_max_pool(bc01, pool_shape, pool_stride, image_shape):
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    wide_infinity = np.ones((bc01.shape[0], bc01.shape[1], required_r, required_c)) * -np.inf
    #import ipdb
    #ipdb.set_trace()
    wide_infinity[:,:, 0:r, 0:c] = bc01
    bc01 = wide_infinity

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            print 'j', cur.shape, row_within_pool, row_stop
            if mx is None:
                mx = cur
            else:
                #import ipdb
                #ipdb.set_trace()
                #p = rng.multinomial(pvals = cur)

                mx = np.maximum(mx, cur)


    return mx


def max_pool(bc01, pool_shape, pool_stride, image_shape):
    """
    Theano's max pooling op only support pool_stride = pool_shape
    so here we have a graph that does max pooling with strides

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    # Compute index in pooled space of last needed pool
    # (needed = each input pixel must appear in at least one pool)
    def last_pool(im_shp, p_shp, p_strd):
        rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
        assert p_strd * rval + p_shp >= im_shp
        assert p_strd * (rval - 1) + p_shp < im_shp
        return rval
    # Compute starting row of the last pool
    last_pool_r = last_pool(image_shape[0] ,pool_shape[0], pool_stride[0]) * pool_stride[0]
    # Compute number of rows needed in image for all indexes to work out
    required_r = last_pool_r + pr

    last_pool_c = last_pool(image_shape[1] ,pool_shape[1], pool_stride[1]) * pool_stride[1]
    required_c = last_pool_c + pc

    for bc01v in get_debug_values(bc01):
        assert not np.any(np.isinf(bc01v))
        assert bc01v.shape[2] == image_shape[0]
        assert bc01v.shape[3] == image_shape[1]

    wide_infinity = T.alloc(-np.inf, bc01.shape[0], bc01.shape[1], required_r, required_c)
    print 'rr', required_r, required_c, last_pool_r, last_pool_c

    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:,:, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    for row_within_pool in xrange(pool_shape[0]):
        row_stop = last_pool_r + row_within_pool + 1
        for col_within_pool in xrange(pool_shape[1]):
            col_stop = last_pool_c + col_within_pool + 1
            cur = bc01[:,:,row_within_pool:row_stop:rs, col_within_pool:col_stop:cs]
            cur.name = 'max_pool_cur_'+bc01.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)
            if mx is None:
                mx = cur
            else:
                #import ipdb
                #ipdb.set_trace()
                #p = rng.multinomial(pvals = cur)

                mx = T.maximum(mx, cur)
                mx.name = 'max_pool_mx_'+bc01.name+'_'+str(row_within_pool)+'_'+str(col_within_pool)

    mx.name = 'max_pool('+name+')'

    for mxv in get_debug_values(mx):
        assert not np.any(np.isnan(mxv))
        assert not np.any(np.isinf(mxv))

    return mx



if __name__ == "__main__":
    theano.config.compute_test_value = 'warn'
    im_shape = (100, 100)
    pool_shape = (3, 3)
    pool_stride = (2, 2)
    x = tensor.tensor4()
    inp = numpy.random.random((100, 20, im_shape[0], im_shape[1])).astype('float32') + 1
    #inp = numpy.asarray([[1.6, 0, 0, 3.0], [0,1.2,0, 0.1],[0, 0.2,0,2.4]]).astype('float32')
    #inp = inp.reshape((1,1,3,4))
    #inp = numpy.asarray([[1.6, 0, 0], [0,0,0],[0,0,2.4]]).astype('float32')
    #inp = inp.reshape((1,1,3,3))
    x.tag.test_value = inp


    # new 1
    start = time.time()
    y = stochastic_max_pool(x, pool_shape, pool_stride, im_shape)
    f = theano.function([x], y)
    print '\nold method compile time: {}'.format((time.time() -start) / 60.)

    start = time.time()
    old = f(inp)
    print 'old method run time: {}'.format((time.time() -start) / 60.)



    start = time.time()
    y = stochastic_max_pool2(x, pool_shape, pool_stride, im_shape)
    f = theano.function([x], y)
    print '\nnew method compile time: {}'.format((time.time() -start) / 60.)

    start = time.time()
    new = f(inp)
    print 'new method run time: {}'.format((time.time() -start) / 60.)


    assert new.shape == old.shape
    assert numpy.array_equal(new, old)
    #for i in xrange(20):
    #print f(inp)

    #x = T.tensor4()
    #y = max_pool(x, (4,4), (3,1), (20,20))
    #f = theano.function([x], y)
    #inp = np.random.random((1,1,20, 20)).astype('float32')
    #res = f(inp)
    #print res.shape

    #npr = numpy_max_pool(inp, (4,4), (4,4), (20,20))
    #assert np.array_equal(npr, res)

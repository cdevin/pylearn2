"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from collections import OrderedDict
import numpy as np
import warnings

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T
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



def stochastic_max_pool(bc01, pool_shape, pool_stride, image_shape, rng):
    """
    Stochastic Pooling for Regularization of Deep Convolutional Neural Networks
    Matthew D. Zeiler, Rob Fergus

    bc01: minibatch in format (batch size, channels, rows, cols)
    pool_shape: shape of the pool region (rows, cols)
    pool_stride: strides between pooling regions (row stride, col stride)
    image_shape: avoid doing some of the arithmetic in theano
    """
    mx = None
    r, c = image_shape
    pr, pc = pool_shape
    rs, cs = pool_stride

    batch = bc01.shape[0]
    channel = bc01.shape[1]

    if rng is None:
        rng = T.shared_randomstreams.RandomStreams(2022)

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

    wide_infinity = T.alloc(0.0, batch, channel, required_r, required_c)

    name = bc01.name
    if name is None:
        name = 'anon_bc01'
    bc01 = T.set_subtensor(wide_infinity[:,:, 0:r, 0:c], bc01)
    bc01.name = 'infinite_padded_' + name

    res_r = int(np.floor(last_pool_r/rs)) + 1
    res_c = int(np.floor(last_pool_c/cs)) + 1
    res = T.alloc(0.0, batch, channel, res_r, res_c)

    for r in xrange(res_r):
        for c in xrange(res_c):
            window = bc01[:, :, r*rs:r*rs+pr, c*cs:c*cs+pc]
            window = window / window.sum(axis = [2, 3]).dimshuffle(0, 1, 'x', 'x')
            prob = rng.multinomial(pvals = window.reshape((batch, channel, pr * pc)))
            val = (bc01[:,:,r*rs:r*rs+pr, c*cs:c*cs+pc] * prob.reshape((batch, channel, pr, pc))).max(axis=[2, 3])
            res = T.set_subtensor(res[:,:,r,c], val)

    return res

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
            # find the max prob

            # select val


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
    #theano.config.compute_test_value = 'warn'
    x = T.tensor4()
    inp = np.random.random((20,3, 20, 20)).astype('float32') + 1
    #inp = np.asarray([[1.6, 0, 0], [0,0,0],[0,0,2.4]]).astype('float32')
    #inp = inp.reshape((1,1,3,3))
    #x.tag.test_value = inp
    y = stochastic_max_pool(x, (3,3), (3,3), (20,20))
    f = theano.function([x], y)
    print 'complied'
    #inp = np.random.random((1,1,20, 20)).astype('float32')
    res = f(inp)
    print res.shape
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

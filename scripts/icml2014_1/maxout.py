"""
MLP Layer objects related to the paper

Maxout Networks. Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
Courville, and Yoshua Bengio. ICML 2013.

If you use this code in your research, please cite this paper.

The objects in this module are Layer objects for use with
pylearn2.models.mlp.MLP. You need to make an MLP object in
order for thse to do anything. For an example of how to build
an MLP with maxout hidden layers, see pylearn2/scripts/papers/maxout.

Note that maxout is designed for use with dropout, so you really should
use dropout in your MLP when using these layers.

Note to developers / maintainers: when making changes to this module,
ensure that the changes do not break the examples in
pylearn2/scripts/papers/maxout.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.sandbox import cuda
from theano import tensor as T

from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import py_integer_types
from pylearn2.utils import sharedX

from pylearn2.linear.conv2d_c01b import setup_detector_layer_c01b
if cuda.cuda_available:
    from pylearn2.sandbox.cuda_convnet.pool import max_pool_c01b
from pylearn2.linear import local_c01b
from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.models.maxout import Maxout


class Maxout2(Maxout):

    def set_input_space(self, space):

        super(Maxout2, self).set_input_space(space)
        self.output_space = (VectorSpace(self.pool_layer_dim / 2), VectorSpace((self.pool_layer_dim -1)/ 2))

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.detector_layer_dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.detector_layer_dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.detector_layer_dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.detector_layer_dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)


    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below) + self.b

        if not hasattr(self, 'randomize_pools'):
            self.randomize_pools = False

        if not hasattr(self, 'pool_stride'):
            self.pool_stride = self.pool_size

        if self.randomize_pools:
            z = T.dot(z, self.permute)

        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        if self.min_zero:
            p_1 = T.zeros_like(z)
            p_2 = T.zeros_like(z)
        else:
            p = None
            p_2 = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(0,self.pool_size, 2):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p_1 = cur
            else:
                p_1 = T.maximum(cur, p)

        p_1.name = self.layer_name + '_p_1_'

        last_start +=1
        for i in xrange(1, self.pool_size, 2):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p_2 = cur
            else:
                p_2 = T.maximum(cur, p)

        p_2.name = self.layer_name + '_p_2_'



        return p_1, p_2



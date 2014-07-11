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
from pylearn2.models.maxout import Maxout, MaxoutConvC01B

class WeightedMaxout(Maxout):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013
    """

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
            p = T.zeros_like(z)
        else:
            p = None

        p_max = None
        p_sum = None
        p_sm = None

        last_start = self.detector_layer_dim  - self.pool_size
        for i in xrange(self.pool_size):
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p_max is None:
                p_max = cur
            else:
                p_max = T.maximum(cur, p_max)

        for i in xrange(self.pool_size):
            cur = T.exp(z[:,i:last_start+i+1:self.pool_stride] - p_max)
            if p_sum is None:
                p_sum = cur
            else:
                p_sum = cur + p_sum

        p_sum = T.switch(T.eq(p_sum, 0), 1e-10, p_sum)
        for i in xrange(self.pool_size):
            cur_ = T.exp(z[:,i:last_start+i+1:self.pool_stride] - p_max)
            cur = z[:,i:last_start+i+1:self.pool_stride]
            if p is None:
                p_sm = (cur_ / p_sum)
                p = cur
            else:
                p_ = (cur_ / p_sum)
                sg = T.gt(p_, p_sm)
                sg_ = T.switch(sg, 0, 1)
                p_sm = p_ * sg + sg_ * p_sm
                p = cur * sg + sg_ * p
                #p = T.concatenate([(cur / p_sum).dimshuffle(0, 1, 'x'), p], axis=2)

        #p = T.argmax(p, axis = 2)

        #import ipdb
        #ipdb.set_trace()


        p.name = self.layer_name + '_p_'

        return p

class WeightedMaxoutConvC01B(MaxoutConvC01B):
    """
    Maxout units arranged in a convolutional layer, with
    spatial max pooling on top of the maxout. If you use this
    code in a research project, please cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013


    This uses the C01B ("channels", topological axis 0,
    topological axis 1, "batch") format of tensors for input
    and output.

    The back-end is Alex Krizhevsky's cuda-convnet library,
    so it is extremely fast, but requires a GPU.
    """


    def fprop(self, state_below):


        def weight_max(z, num_pieces):

            p = None
            p_max = None
            p_sum = None
            p_sm = None

            for i in xrange(num_pieces):
                cur = z[i::num_pieces,:,:,:]
                if p_max is None:
                    p_max = cur
                else:
                    p_max = T.maximum(cur, p_max)

            for i in xrange(num_pieces):
                cur = T.exp(z[i::num_pieces,:,:,:] - p_max)
                if p_sum is None:
                    p_sum = cur
                else:
                    p_sum = cur + p_sum

            #p_sum = T.switch(T.eq(p_sum, 0), 1e-10, p_sum)
            for i in xrange(num_pieces):
                cur_ = T.exp(z[i::num_pieces,:,:,:] - p_max)
                cur = z[i::num_pieces,:,:,:]
                if p is None:
                    p_sm = (cur_ / p_sum)
                    p = cur
                else:
                    p_ = (cur_ / p_sum)
                    sg = T.gt(p_, p_sm)
                    sg_ = T.switch(sg, 0, 1)
                    p_sm = p_ * sg + sg_ * p_sm
                    p = cur * sg + sg_ * p


            ####
            #s = None
            #for i in xrange(num_pieces):
                #t = z[i::num_pieces,:,:,:]
                #if s is None:
                    #s = t
                #else:
                    #s = T.maximum(s, t)
            #return s
            return p

        check_cuda(str(type(self)))

        self.input_space.validate(state_below)

        if not hasattr(self, 'input_normalization'):
            self.input_normalization = None

        if self.input_normalization:
            state_below = self.input_normalization(state_below)

        # Alex's code requires # input channels to be <= 3 or a multiple of 4
        # so we add dummy channels if necessary
        if not hasattr(self, 'dummy_channels'):
            self.dummy_channels = 0
        if self.dummy_channels > 0:
            state_below = T.concatenate((state_below,
                                         T.zeros_like(state_below[0:self.dummy_channels, :, :, :])),
                                        axis=0)

        z = self.transformer.lmul(state_below)
        if not hasattr(self, 'tied_b'):
            self.tied_b = False
        if self.tied_b:
            b = self.b.dimshuffle(0, 'x', 'x', 'x')
        else:
            b = self.b.dimshuffle(0, 1, 2, 'x')


        z = z + b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        self.detector_space.validate(z)

        assert self.detector_space.num_channels % 16 == 0

        if self.output_space.num_channels % 16 == 0:
            # alex's max pool op only works when the number of channels
            # is divisible by 16. we can only do the cross-channel pooling
            # first if the cross-channel pooling preserves that property
            if self.num_pieces != 1:
                z = weight_max(z, self.num_pieces)

            if self.detector_normalization:
                z = self.detector_normalization(z)

            p = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            z = max_pool_c01b(c01b=z, pool_shape=self.pool_shape,
                              pool_stride=self.pool_stride,
                              image_shape=self.detector_space.shape)
            if self.num_pieces != 1:
                z = weight_max(z, self.num_pieces)
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p




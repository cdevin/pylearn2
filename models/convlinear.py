"""
Multilayer Perceptron
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

from collections import OrderedDict
import numpy as numpy
import warnings
import sys

from theano import config
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano import tensor as T

from pylearn2.linear import conv2d
try:
    from pylearn2.linear import conv2d_c01b
except ImportError:
    warnings.warn("Couldn't import Alex-style convolution, probably because you don't have a GPU"
            "Some stuff might be broken.")
from pylearn2.models.mlp import MLP
from pylearn2.utils import safe_izip
from galatea.mlp import ConvLinearC01B
from noisy_encoder.models.conv import stochastic_softmax_pool, softmax_pool


class MLP(MLP):
    def fprop(self, state_below, apply_dropout = False, return_all = False, train_prop = False):

        if apply_dropout:
            warnings.warn("dropout should be implemented with fixed_var_descr to make sure it works with BGD, this is just a hack to get it working with SGD")
            theano_rng = MRG_RandomStreams(self.rng.randint(2**15))
            scale = self.dropout_input_scale
            state_below = self.apply_dropout(state=state_below,
                    include_prob=self.dropout_input_include_prob,
                    theano_rng=theano_rng,
                    scale=scale)

        layer =  self.layers[0]
        if isinstance(layer, ConvLinearC01BStochastic):
            rval = layer.fprop(state_below, train_prop)
        else:
            rval = layer.fprop(state_below)


        if apply_dropout:
            dropout = self.dropout_include_probs[0]
            scale = self.dropout_scales[0]
            rval = self.apply_dropout(state=rval, include_prob=dropout, theano_rng=theano_rng,
                    scale=scale)
        rlist = [rval]

        for layer, dropout, scale in safe_izip(self.layers[1:], self.dropout_include_probs[1:],
            self.dropout_scales[1:]):
            if isinstance(layer, ConvLinearC01BStochastic):
                rval = layer.fprop(rval, train_prop)
            else:
                rval = layer.fprop(rval)
            if apply_dropout:
                rval = self.apply_dropout(state=rval, include_prob=dropout, theano_rng=theano_rng,
                        scale=scale)
            rlist.append(rval)

        if return_all:
            return rlist
        return rval

    def cost_from_X(self, X, Y):
        Y_hat = self.fprop(X, apply_dropout = self.use_dropout, train_prop = True)
        return self.cost(Y, Y_hat)

class ConvLinearStochasticSoftmaxPoolC01B(ConvLinearC01B):
    """
    """
    def fprop(self, state_below, train_prop = False):

        self.input_space.validate(state_below)

        state_below = self.input_space.format_as(state_below, self.desired_space)

        # check if it's train pass or test pass
        if train_prop:
            pool_f = getattr(sys.modules[__name__], 'stochastic_softmax_pool_c01b')
        else:
            pool_f = getattr(sys.modules[__name__], 'softmax_pool_c01b')

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
            if self.channel_pool_size != 1:
                s = None
                for i in xrange(self.channel_pool_size):
                    t = z[i::self.channel_pool_size,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            p = pool_f(c01b=z, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        else:
            z = pool_f(c01b=z, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
            if self.channel_pool_size != 1:
                s = None
                for i in xrange(self.channel_pool_size):
                    t = z[i::self.channel_pool_size,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z


        self.output_space.validate(p)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p


def stochastic_softmax_pool_c01b(c01b, pool_shape, pool_stride, image_shape):
    bc01 = c01b.dimshuffle(3,0,1,2)
    rval = stochastic_sofmax_pool(bc01, pool_shape, pool_stride, image_shape)
    return rval.dimshuffle(1, 2, 3, 0)

def softmax_pool_c01b(c01b, pool_shape, pool_stride, image_shape):
    bc01 = c01b.dimshuffle(3,0,1,2)
    rval = softmax_pool(bc01, pool_shape, pool_stride, image_shape)
    return rval.dimshuffle(1, 2, 3, 0)


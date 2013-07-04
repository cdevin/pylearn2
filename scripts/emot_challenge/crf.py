import math
import sys
import warnings

import numpy as np
from theano import config
from theano.compat.python2x import OrderedDict
from theano.gof.op import get_debug_values
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano.tensor as T

from pylearn2.costs.mlp import Default
from pylearn2.expr.probabilistic_max_pooling import max_pool_channels
from pylearn2.linear import conv2d
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.expr.nnet import pseudoinverse_softmax_numpy
from pylearn2.space import CompositeSpace
from pylearn2.space import Conv2DSpace
from pylearn2.space import Space
from pylearn2.space import VectorSpace
from pylearn2.utils import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import safe_union
from pylearn2.utils import safe_zip
from pylearn2.utils import sharedX


class Softmax(Layer):

    def __init__(self, n_classes, layer_name, irange = None,
            istdev = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False,
                 max_col_norm = None, init_bias_target_marginals= None):
        """
        """

        if isinstance(W_lr_scale, str):
            W_lr_scale = float(W_lr_scale)

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        self.output_space = VectorSpace(n_classes)
        if not no_affine:
            self.b = sharedX( np.zeros((n_classes,)), name = 'softmax_b')
            if init_bias_target_marginals:
                marginals = init_bias_target_marginals.y.mean(axis=0)
                assert marginals.ndim == 1
                b = pseudoinverse_softmax_numpy(marginals).astype(self.b.dtype)
                assert b.ndim == 1
                assert b.dtype == self.b.dtype
                self.b.set_value(b)
        else:
            assert init_bias_target_marginals is None

    def get_lr_scalers(self):

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            assert isinstance(self.W_lr_scale, float)
            rval[self.W] = self.W_lr_scale

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        if self.b_lr_scale is not None:
            assert isinstance(self.b_lr_scale, float)
            rval[self.b] = self.b_lr_scale

        return rval

    def get_monitoring_channels(self):

        if self.no_affine:
            return OrderedDict()

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            ])

    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)

        return rval

    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got "+
                    str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng

        if self.no_affine:
            self._params = []
        else:
            if self.irange is not None:
                assert self.istdev is None
                assert self.sparse_init is None
                W = rng.uniform(-self.irange,self.irange, (self.input_dim,self.n_classes))
            elif self.istdev is not None:
                assert self.sparse_init is None
                W = rng.randn(self.input_dim, self.n_classes) * self.istdev
            else:
                assert self.sparse_init is not None
                W = np.zeros((self.input_dim, self.n_classes))
                for i in xrange(self.n_classes):
                    for j in xrange(self.sparse_init):
                        idx = rng.randint(0, self.input_dim)
                        while W[idx, i] != 0.:
                            idx = rng.randint(0, self.input_dim)
                        W[idx, i] = rng.randn()

            self.W = sharedX(W,  'softmax_W' )

            self._params = [ self.b, self.W ]

    def get_weights_topo(self):
        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()
        desired = self.W.get_value().T
        ipt = self.desired_space.format_as(desired, self.input_space)
        rval = Conv2DSpace.convert_numpy(ipt, self.input_space.axes, ('b', 0, 1, 'c'))
        return rval

    def get_weights(self):
        if not isinstance(self.input_space, VectorSpace):
            raise NotImplementedError()

        return self.W.get_value()

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        for value in get_debug_values(state_below):
            if self.mlp.batch_size is not None and value.shape[0] != self.mlp.batch_size:
                raise ValueError("state_below should have batch size "+str(self.dbm.batch_size)+" but has "+str(value.shape[0]))

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        if self.no_affine:
            Z = state_below
        else:
            assert self.W.ndim == 2
            b = self.b

            Z = T.dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            Y_hat, = owner.inputs
            owner = Y_hat.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z ,= owner.inputs
        assert z.ndim == 2

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval

    def get_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        return coeff * T.sqr(self.W).sum()

    def get_l1_weight_decay(self, coeff):
        if isinstance(coeff, str):
            coeff = float(coeff)
        assert isinstance(coeff, float) or hasattr(coeff, 'dtype')
        W = self.W
        return coeff * abs(W).sum()

    def censor_updates(self, updates):
        if self.no_affine:
            return
        if self.max_row_norm is not None:
            W = self.W
            if W in updates:
                updated_W = updates[W]
                row_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=1))
                desired_norms = T.clip(row_norms, 0, self.max_row_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + row_norms)).dimshuffle(0, 'x')
        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W = self.W
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

class CompositeLayer(Layer):
    """
    A Layer that runs several simpler layers in parallel.
    """

    def __init__(self, layer_name, layers):
        """
        layers: a list or tuple of Layers.
        """
        self.__dict__.update(locals())
        del self.self

    def set_input_space(self, space):

        for layer in self.layers:
            layer.set_input_space(space)

        self.output_space = CompositeSpace(tuple(layer.get_output_space()
            for layer in self.layers))

    def get_params(self):

        rval = []

        for layer in self.layers:
            rval = safe_union(layer.get_params(), rval)

        return rval

    def fprop(self, state_below):

        return tuple(layer.fprop(state_below) for layer in self.layers)

    def cost(self, Y, Y_hat):

        return sum(layer.cost(Y_elem, Y_hat_elem) for layer, Y_elem, Y_hat_elem in \
                safe_zip(self.layers, Y, Y_hat))

    def set_mlp(self, mlp):
        super(CompositeLayer, self).set_mlp(mlp)
        for layer in self.layers:
            layer.set_mlp(mlp)



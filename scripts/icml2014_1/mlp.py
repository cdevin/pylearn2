"""
Multilayer Perceptron

Note to developers and code reviewers: when making any changes to this
module, ensure that the changes do not break pylearn2/scripts/papers/maxout.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2012-2013, Universite de Montreal"
__credits__ = ["Ian Goodfellow", "David Warde-Farley"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"

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

warnings.warn("MLP changing the recursion limit.")
# We need this to be high enough that the big theano graphs we make
# when doing max pooling via subtensors don't cause python to complain.
# python intentionally declares stack overflow well before the stack
# segment is actually exceeded. But we can't make this value too big
# either, or we'll get seg faults when the python interpreter really
# does go over the stack segment.
# IG encountered seg faults on eos3 (a machine at LISA labo) when using
# 50000 so for now it is set to 40000.
# I think the actual safe recursion limit can't be predicted in advance
# because you don't know how big of a stack frame each function will
# make, so there is not really a "correct" way to do this. Really the
# python interpreter should provide an option to raise the error
# precisely when you're going to exceed the stack segment.
sys.setrecursionlimit(40000)


class MLP(Layer):
    """
    A multilayer perceptron.
    Note that it's possible for an entire MLP to be a single
    layer of a larger MLP.
    """

    def __init__(self,
            layers,
            batch_size=None,
            input_space=None,
            nvis=None,
            seed=None,
            dropout_include_probs = None,
            dropout_scales = None,
            dropout_input_include_prob = None,
            dropout_input_scale = None,
            ):
        """
            layers: a list of MLP_Layers. The final layer will specify the
                    MLP's output space.
            batch_size: optional. if not None, then should be a positive
                        integer. Mostly useful if one of your layers
                        involves a theano op like convolution that requires
                        a hard-coded batch size.
            input_space: a Space specifying the kind of input the MLP acts
                        on. If None, input space is specified by nvis.
            dropout*: None of these arguments are supported anymore. Use
                      pylearn2.costs.mlp.dropout.Dropout instead.
        """

        locals_snapshot = locals()

        for arg in locals_snapshot:
            if arg.find('dropout') != -1 and locals_snapshot[arg] is not None:
                raise TypeError(arg+ " is no longer supported. Train using "
                        "an instance of pylearn2.costs.mlp.dropout.Dropout "
                        "instead of hardcoding the dropout into the model"
                        " itself. All dropout related arguments and this"
                        " support message may be removed on or after "
                        "October 2, 2013. They should be removed from the "
                        "SoftmaxRegression subclass at the same time.")

        if seed is None:
            seed = [2013, 1, 4]

        self.seed = seed
        self.setup_rng()

        assert isinstance(layers, list)
        assert all(isinstance(layer, Layer) for layer in layers)
        assert len(layers) >= 1
        self.layer_names = set()
        for layer in layers:
            if isinstance(layer, tuple):
                for item in layer:
                    assert layer.get_mlp() is None
                    assert layer.layer_name not in self.layer_names
                    layer.set_mlp(self)
                    self.layer_names.add(layer.layer_name)
            else:
                assert layer.get_mlp() is None
                assert layer.layer_name not in self.layer_names
                layer.set_mlp(self)
                self.layer_names.add(layer.layer_name)

        self.layers = layers

        self.batch_size = batch_size
        self.force_batch_size = batch_size

        assert input_space is not None or nvis is not None
        if nvis is not None:
            input_space = VectorSpace(nvis)

        self.input_space = input_space

        self._update_layer_input_spaces()

        self.freeze_set = set([])

        def f(x):
            if x is None:
                return None
            return 1. / x

    def setup_rng(self):
        self.rng = np.random.RandomState(self.seed)

    def get_default_cost(self):
        return Default()

    def get_output_space(self):
        return self.layers[-1].get_output_space()

    def _update_layer_input_spaces(self):
        """
            Tells each layer what its input space should be.
            Note: this usually resets the layer's parameters!
        """
        layers = self.layers
        layers[0].set_input_space(self.input_space)
        for i in xrange(1,len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

    def add_layers(self, layers):
        """
            Add new layers on top of the existing hidden layers
        """

        existing_layers = self.layers
        assert len(existing_layers) > 0
        for layer in layers:
            assert layer.get_mlp() is None
            layer.set_mlp(self)
            layer.set_input_space(existing_layers[-1].get_output_space())
            existing_layers.append(layer)
            assert layer.layer_name not in self.layer_names
            self.layer_names.add(layer.layer_name)

    def freeze(self, parameter_set):

        self.freeze_set = self.freeze_set.union(parameter_set)

    def get_monitoring_channels(self, data):
        """
        data is a flat tuple, and can contain features, targets, or both
        """
        X, Y = data
        state = X
        rval = OrderedDict()

        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            state = layer.fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

        return rval

    def get_monitoring_data_specs(self):
        """
        Return the (space, source) data_specs for self.get_monitoring_channels.

        In this case, we want the inputs and targets.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)

    def get_params(self):

        rval = []
        for layer in self.layers:
            for param in layer.get_params():
                if param.name is None:
                    print type(layer)
            layer_params = layer.get_params()
            assert not isinstance(layer_params, set)
            for param in layer_params:
                if param not in rval:
                    rval.append(param)

        rval = [elem for elem in rval if elem not in self.freeze_set]

        assert all([elem.name is not None for elem in rval])

        return rval

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.force_batch_size = batch_size

        for layer in self.layers:
            layer.set_batch_size(batch_size)


    def censor_updates(self, updates):
        for layer in self.layers:
            layer.censor_updates(updates)

    def get_lr_scalers(self):
        rval = OrderedDict()

        params = self.get_params()

        for layer in self.layers:
            contrib = layer.get_lr_scalers()

            assert isinstance(contrib, OrderedDict)
            # No two layers can contend to scale a parameter
            assert not any([key in rval for key in contrib])
            # Don't try to scale anything that's not a parameter
            assert all([key in params for key in contrib])

            rval.update(contrib)
        assert all([isinstance(val, float) for val in rval.values()])

        return rval

    def get_weights(self):
        return self.layers[0].get_weights()

    def get_weights_view_shape(self):
        return self.layers[0].get_weights_view_shape()

    def get_weights_format(self):
        return self.layers[0].get_weights_format()

    def get_weights_topo(self):
        return self.layers[0].get_weights_topo()

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):
        """
        state_below: The input to the MLP

        Returns the output of the MLP, when applying dropout to the input and intermediate layers.
        Each input to each layer is randomly included or excluded
        for each example. The probability of inclusion is independent for each input
        and each example. Each layer uses "default_input_include_prob" unless that
        layer's name appears as a key in input_include_probs, in which case the input
        inclusion probability is given by the corresponding value.

        Each feature is also multiplied by a scale factor. The scale factor for each
        layer's input scale is determined by the same scheme as the input probabilities.

        per_example : bool, optional
            Sample a different mask value for every example in
            a batch. Default is `True`. If `False`, sample one
            mask per mini-batch.
        """

        warnings.warn("dropout doesn't use fixed_var_descr so it won't work with "
                "algorithms that make more than one theano function call per batch,"
                " such as BGD. Implementing fixed_var descr could increase the memory"
                " usage though.")

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(self.rng.randint(2 ** 15))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )
            state_below = layer.fprop(state_below)

        return state_below

    def _validate_layer_names(self, layers):
        if any(layer not in self.layer_names for layer in layers):
            unknown_names = [layer for layer in layers
                                if layer not in self.layer_names]
            raise ValueError("MLP has no layer(s) named %s" %
                                ", ".join(unknown_names))

    def get_total_input_dimension(self, layers):
        """
        Get the total number of inputs to the layers whose
        names are listed in `layers`. Used for computing the
        total number of dropout masks.
        """
        self._validate_layer_names(layers)
        total = 0
        for layer in self.layers:
            if layer.layer_name in layers:
                total += layer.get_input_space().get_total_dimension()
        return total

    def fprop(self, state_below, return_all = False):

        rval = self.layers[0].fprop(state_below)


        for layer in self.layers[1:-2]:
            rval = layer.fprop(rval)

        assert isinstance(self.layers[-1], tuple)

        rvals = []
        for item in self.layers[-1]:
            rvals.append(item.fprop(rval))

        return rvals

    def apply_dropout(self, state, include_prob, scale, theano_rng,
                      input_space, mask_value=0, per_example=True):

        """
        Parameters
        ----------
        ...

        per_example : bool, optional
            Sample a different mask value for every example in
            a batch. Default is `True`. If `False`, sample one
            mask per mini-batch.
        """
        if include_prob in [None, 1.0, 1]:
            return state
        assert scale is not None
        if isinstance(state, tuple):
            return tuple(self.apply_dropout(substate, include_prob,
                                            scale, theano_rng, mask_value)
                         for substate in state)
        # TODO: all of this assumes that if it's not a tuple, it's
        # a dense tensor. It hasn't been tested with sparse types.
        # A method to format the mask (or any other values) as
        # the given symbolic type should be added to the Spaces
        # interface.
        if per_example:
            mask = theano_rng.binomial(p=include_prob, size=state.shape,
                                       dtype=state.dtype)
        else:
            batch = input_space.get_origin_batch(1)
            mask = theano_rng.binomial(p=include_prob, size=batch.shape,
                                       dtype=state.dtype)
            rebroadcast = T.Rebroadcast(*zip(xrange(batch.ndim),
                                             [s == 1 for s in batch.shape]))
            mask = rebroadcast(mask)
        if mask_value == 0:
            return state * mask * scale
        else:
            return T.switch(mask, state * scale, mask_value)

    def cost(self, Ys, Y_hats):
        assert isinstance(Y_hats, tuple)
        assert isinstance(Ys, tuple)
        assert len(Ys) == len(Y_hats)
        assert len(self.layers[-1]) == len(Ys)

        sum = 0
        for i in xrange(len(Ys)):
            sum += self.layers[-1][i].cost(Ys[i], Y_hat[i])

    def cost_matrix(self, Y, Y_hat):
        return self.layers[-1].cost_matrix(Y, Y_hat)

    def cost_from_cost_matrix(self, cost_matrix):
        return self.layers[-1].cost_from_cost_matrix(cost_matrix)

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hats= self.fprop(X)
        return self.cost(Y, Y_hats)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)



from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.gof.op import get_debug_values
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.models.mlp import Layer, MLP, Softmax
from pylearn2.space import Conv2DSpace
from pylearn2.space import VectorSpace
from pylearn2.space import Space
from pylearn2.space import CompositeSpace
from pylearn2.format.target_format import OneHotFormatter
from pylearn2.utils import sharedX


class MLP_GatedRectifier(Layer):
    """
    A hidden layer that does max pooling over groups of linear
    units. If you use this code in a research project, please
    cite

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. arXiv 2013
    """

    def __init__(self,
                 layer_name,
                 dim,
                 gater = None,
                 pool_stride = None,
                 randomize_pools = False,
                 irange = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 sparsity_type = 'kl',
                 sparsity_ratio = 0.3,
                 sparsity_momentum = 0.9,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 max_col_norm = None,
                 max_row_norm = None,
                 mask_weights = None,
        ):
        """
            layer_name: A name for this layer that will be prepended to
                        monitoring channels related to this layer.
            dim: The number of hidden units.
            pool_stride: The distance between the start of each max pooling
                        region. Defaults to num_pieces, which makes the
                        pooling regions disjoint. If set to a smaller number,
                        can do overlapping pools.
            randomize_pools: Does max pooling over randomized subsets of
                        the linear responses, rather than over sequential
                        subsets.
            irange: if specified, initializes each weight randomly in
                U(-irange, irange)
            sparse_init: if specified, irange must not be specified.
                        This is an integer specifying how many weights to make
                        non-zero. All non-zero weights will be initialized
                        randomly in N(0, sparse_stdev^2)
            include_prob: probability of including a weight element in the set
               of weights initialized to U(-irange, irange). If not included
               a weight is initialized to 0. This defaults to 1.
            init_bias: All biases are initialized to this number
            W_lr_scale: The learning rate on the weights for this layer is
                multiplied by this scaling factor
            b_lr_scale: The learning rate on the biases for this layer is
                multiplied by this scaling factor
            max_col_norm: The norm of each column of the weight matrix is
                constrained to have at most this norm. If unspecified, no
                constraint. Constraint is enforced by re-projection (if
                necessary) at the end of each update.
            max_row_norm: Like max_col_norm, but applied to the rows.
            mask_weights: A binary matrix multiplied by the weights after each
                         update, allowing you to restrict their connectivity.
            min_zero: If true, includes a zero in the set we take a max over
                    for each maxout unit. This is equivalent to pooling over
                    rectified linear units.
        """

        self.__dict__.update(locals())
        del self.self

        self.b = sharedX( np.zeros((self.dim,)) + init_bias, name = layer_name + '_b')


        if max_row_norm is not None:
            raise NotImplementedError()

    def get_lr_scalers(self):

        if not hasattr(self, 'W_lr_scale'):
            self.W_lr_scale = None

        if not hasattr(self, 'b_lr_scale'):
            self.b_lr_scale = None

        rval = OrderedDict()

        if self.W_lr_scale is not None:
            W, = self.transformer.get_params()
            rval[W] = self.W_lr_scale

        if self.b_lr_scale is not None:
            rval[self.b] = self.b_lr_scale

        return rval

    def set_input_space(self, space):
        """ Note: this resets parameters! """

        self.input_space = space

        assert self.gater.get_input_space() == space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim, sparse = True)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.dim))
                 < self.include_prob)
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.input_dim, self.dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.dim):
                assert self.sparse_init <= self.input_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.input_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.input_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'
        self.W = W

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)


        self.theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))

    def censor_updates(self, updates):

        # Patch old pickle files
        if not hasattr(self, 'mask_weights'):
            self.mask_weights = None

        if self.mask_weights is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updates[W] = updates[W] * self.mask

        if self.max_col_norm is not None:
            assert self.max_row_norm is None
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms, 0, self.max_col_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + col_norms))

    def get_params(self):
        assert self.b.name is not None
        assert self.W.name is not None
        rval = [self.W, self.b]
        return rval + self.gater.get_params()

    def get_weights(self):

        print 'Which weights? '
        print 'g) gater'
        print 'm) main network'

        x = raw_input()

        if x == 'g':
            return self.gater.get_weights()
        assert x == 'm'

        if self.requires_reformat:
            # This is not really an unimplemented case.
            # We actually don't know how to format the weights
            # in design space. We got the data in topo space
            # and we don't have access to the dataset
            raise NotImplementedError()
        W = self.W.get_value()

        print W.shape

        return W

    def set_weights(self, weights):
        self.W.set_value(weights)

    def set_biases(self, biases):
        self.b.set_value(biases)

    def get_biases(self):
        return self.b.get_value()

    def get_weights_format(self):
        return ('v', 'h')

    def get_weights_topo(self):

        if not isinstance(self.input_space, Conv2DSpace):
            raise NotImplementedError()

        # There was an implementation of this, but it was broken
        raise NotImplementedError()

    def get_monitoring_channels(self):

        W = self.W

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        #var = self.gater.layers[-1].W
        #var = T.argmax(var, axis=1).std().astype(theano.config.floatX)
        return OrderedDict([
                            ('row_norms_min'  , row_norms.min()),
                            ('row_norms_mean' , row_norms.mean()),
                            ('row_norms_max'  , row_norms.max()),
                            ('col_norms_min'  , col_norms.min()),
                            ('col_norms_mean' , col_norms.mean()),
                            ('col_norms_max'  , col_norms.max()),
                            #('softmax_weights_std', var),
                            ])


    #def get_monitoring_channels_from_state(self, state):

        #P = state

        #rval = OrderedDict()

        #vars_and_prefixes = [ (P,'') ]

        #for var, prefix in vars_and_prefixes:
            #v_max = var.max(axis=0)
            #v_min = var.min(axis=0)
            #v_mean = var.mean(axis=0)
            #v_range = v_max - v_min

            ## max_x.mean_u is "the mean over *u*nits of the max over e*x*amples"
            ## The x and u are included in the name because otherwise its hard
            ## to remember which axis is which when reading the monitor
            ## I use inner.outer rather than outer_of_inner or something like that
            ## because I want mean_x.* to appear next to each other in the alphabetical
            ## list, as these are commonly plotted together
            #for key, val in [
                             #('max_x.max_u', v_max.max()),
                             #('max_x.mean_u', v_max.mean()),
                             #('max_x.min_u', v_max.min()),
                             #('min_x.max_u', v_min.max()),
                             #('min_x.mean_u', v_min.mean()),
                             #('min_x.min_u', v_min.min()),
                             #('range_x.max_u', v_range.max()),
                             #('range_x.mean_u', v_range.mean()),
                             #('range_x.min_u', v_range.min()),
                             #('mean_x.max_u', v_mean.max()),
                             #('mean_x.mean_u', v_mean.mean()),
                             #('mean_x.min_u', v_mean.min())
                             #]:
                #rval[prefix+key] = val

        #return rval

    def fprop(self, state_below):

        self.input_space.validate(state_below)

        if self.requires_reformat:
            if not isinstance(state_below, tuple):
                for sb in get_debug_values(state_below):
                    if sb.shape[0] != self.dbm.batch_size:
                        raise ValueError("self.dbm.batch_size is %d but got shape of %d" % (self.dbm.batch_size, sb.shape[0]))
                    assert reduce(lambda x,y: x * y, sb.shape[1:]) == self.input_dim

            state_below = self.input_space.format_as(state_below, self.desired_space)


        if not hasattr(self, 'min_zero'):
            self.min_zero = False

        #z = T.clip(z, 0., 1e30)
        #p = None

        gate = self.gater.fprop(state_below)
        gate = theano.sparse.csr_from_dense(gate)
        #z = theano.sparse.basic.sampling_dot(state_below, self.W.T, gate)
        #b = theano.sparse.basic.mul_s_v(gate, self.b)
        #z = theano.sparse.basic.add_s_s(z, b)
        z = T.dot(state_below, self.W.T) + self.b
        z = theano.sparse.mul_s_d(gate, z)
        z = theano.sparse.basic.structured_sigmoid(z)
        z.name = self.layer_name + '_z_'


        return z

    def cost_sparsity(self, state):

        z = self.gater.fprop(state)
        if self.sparsity_type == 'l1':
            rval = z.sum(axis=1).mean()
        elif self.sparsity_type == 'kl':
            z_old = z.mean(axis=0)
            z = self.sparsity_momentum * z_old + (1. - self.sparsity_momentum) * z
            rval = self.sparsity_ratio * T.log(z) + (1. - self.sparsity_ratio) * T.log(1-z)
            rval = rval.sum(axis=1).mean()
        else:
            raise ValueError("Unknown sparsity type: {}".format(sparsity_type))
        return -rval


class SparseSoftmax(Softmax):

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
        self.desired_space = VectorSpace(desired_dim, sparse = True)

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

    def fprop(self, state_below):

        #import ipdb
        #ipdb.set_trace()
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

            #Z = T.dot(state_below, self.W) + b
            Z = theano.sparse.basic.structured_dot(state_below, self.W) + b

        rval = T.nnet.softmax(Z)

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval

    def get_monitoring_channels(self):
        return OrderedDict([])

    def get_monitoring_channels_from_state(self, state, target=None):
        return OrderedDict([])

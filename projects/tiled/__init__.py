import numpy as np
from theano import tensor as T
from theano.printing import Print
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import Softmax
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear import local_c01b
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.format.target_format import OneHotFormatter

class LocalLinear(Linear):

    def __init__(self,
                 dim,
                 kernel_shape,
                 layer_name,
                 kernel_stride = (1, 1),
                 pad = 0,
                 partial_sum = 1,
                 irange = None,
                 istdev = None,
                 sparse_init = None,
                 sparse_stdev = 1.,
                 include_prob = 1.0,
                 init_bias = 0.,
                 W_lr_scale = None,
                 b_lr_scale = None,
                 mask_weights = None,
                 max_row_norm = None,
                 max_col_norm = None,
                 softmax_columns = False,
                 copy_input = 0,
                 use_abs_loss = False,
                 use_bias = True):

        self.__dict__.update(locals())
        del self.self
        super(LocalLinear, self).__init__(dim = dim,
                 layer_name = layer_name,
                 irange = irange,
                 istdev = istdev,
                 sparse_init = sparse_init,
                 sparse_stdev = sparse_stdev,
                 include_prob = include_prob,
                 init_bias = init_bias,
                 W_lr_scale = W_lr_scale,
                 b_lr_scale = b_lr_scale,
                 mask_weights = mask_weights,
                 max_row_norm = max_row_norm,
                 max_col_norm = max_col_norm,
                 softmax_columns = softmax_columns,
                 copy_input = copy_input,
                 use_abs_loss = use_abs_loss,
                 use_bias = use_bias)

    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, Conv2DSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.desired_space = Conv2DSpace(shape = self.image_shape,
                                            channels = 1,
                                            axes = ('c', 0, 1, 'b'))

        self.output_space = VectorSpace(self.dim + self.copy_input * self.input_dim)

        rng = self.mlp.rng

        # find the image dimension
        image_shape = int(np.sqrt(self.input_space.dim))
        assert image_shape ** 2 == self.input_space.dim
        self.image_shape = (image_shape, image_shape)

        self.transformer = local_c01b.make_random_local(
                input_groups = 1,
                irange = self.irange,
                input_axes = ('c', 0, 1, 'b'),
                image_shape = self.image_shape,
                output_axes =('c', 0, 1, 'b'),
                input_channels = 1,
                output_channels = 1,
                kernel_shape = self.kernel_shape,
                kernel_stride = self.kernel_stride,
                pad = self.pad,
                partial_sum = self.partial_sum,
                rng = rng)


        W ,= self.transformer.get_params()
        W.name = self.layer_name + '_W'

        if self.mask_weights is not None:
            expected_shape =  (self.input_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)

    def _linear_part(self, state_below):
        """
        .. todo::

            WRITEME
        """
        self.input_space.validate(state_below)

        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        z = self.transformer.lmul(state_below).flatten(2) + self.b
        if self.layer_name is not None:
            z.name = self.layer_name + '_z'

        return z



class EmbeddingLinear(Linear):

    def __init__(self, dict_dim = 1000, **kwargs):

        super(EmbeddingLinear, self).__init__(**kwargs)
        self.dict_dim = dict_dim

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME

        Notes
        -----
        This resets parameters!
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim * self.input_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W = rng.uniform(-self.irange,
                            self.irange,
                            (self.dict_dim, self.dim)) * \
                (rng.uniform(0.,1., (self.dict_dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W = rng.randn(self.dict_dim, self.dim) * self.istdev
        else:
            assert self.sparse_init is not None
            W = np.zeros((self.dict_dim, self.dim))
            def mask_rejects(idx, i):
                if self.mask_weights is None:
                    return False
                return self.mask_weights[idx, i] == 0.
            for i in xrange(self.dim):
                assert self.sparse_init <= self.dict_dim
                for j in xrange(self.sparse_init):
                    idx = rng.randint(0, self.dict_dim)
                    while W[idx, i] != 0 or mask_rejects(idx, i):
                        idx = rng.randint(0, self.dict_dim)
                    W[idx, i] = rng.randn()
            W *= self.sparse_stdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.dict_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)



    def _linear_part(self, state_below):


        self.input_space.validate(state_below)
        if self.requires_reformat:
            state_below = self.input_space.format_as(state_below, self.desired_space)

        W, = self.transformer.get_params()
        z = []
        z = W[state_below.flatten().astype('int8')] + self.b
        z = z.reshape((state_below.shape[0], self.input_dim * self.dim))

        return z


class TransposeEmbeddingLinear(Linear):

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME

        Notes
        -----
        This resets parameters!
        """

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng

        W = sharedX(W)


        W, = self.mlp.layers[0].get_params()
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W.T)

        W ,= self.transformer.get_params()
        assert W.name is not None

        if self.mask_weights is not None:
            expected_shape =  (self.dict_dim, self.dim)
            if expected_shape != self.mask_weights.shape:
                raise ValueError("Expected mask with shape "+str(expected_shape)+" but got "+str(self.mask_weights.shape))
            self.mask = sharedX(self.mask_weights)


class CompactSoftmax(Softmax):

    def __init__(self, **kwargs):

        super(CompactSoftmax, self).__init__(**kwargs)
        self.output_space = VectorSpace(1)


    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.

        Parameters
        ----------
        Y : WRITEME
        Y_hat : WRITEME

        Returns
        -------
        WRITEME
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
        Y = OneHotFormatter(self.n_classes).theano_expr(
                                T.addbroadcast(Y, 1).dimshuffle(0).astype('int8'))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval

    def get_monitoring_channels_from_state(self, state, target=None):
        """
        .. todo::

            WRITEME
        """

        mx = state.max(axis=1)

        rval =  OrderedDict([
                ('mean_max_class' , mx.mean()),
                ('max_max_class' , mx.max()),
                ('min_max_class' , mx.min())
        ])

        if target is not None:
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['perplexity'] = 10 ** (rval['nll'] / np.log(10).astype('float32'))

        return rval


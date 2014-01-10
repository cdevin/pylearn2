import numpy as np
from theano import tensor as T
from theano.printing import Print
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import ConvRectifiedLinear
from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import max_pool
from pylearn2.models.mlp import mean_pool
from pylearn2.models.maxout import MaxoutLocalC01B
from pylearn2.linear.matrixmul import MatrixMul
from pylearn2.linear import local_c01b
from pylearn2.space import VectorSpace
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX
from pylearn2.format.target_format import OneHotFormatter

class LocalLinearVector(Linear):

    def __init__(self, dim, layer_name, kernel_shape, kernel_stride = (1, 1), **kwargs):
        super(LocalLinear, self).__init__(dim, layer_name, **kwargs)
        self.kernel_shape = kernel_shape
        self.kernel_stride = kernel_stride

    def set_input_space(self, space):

        self.input_space = space

        # find the image dimension
        image_shape = int(np.sqrt(self.dim))
        assert image_shape ** 2 == self.dim
        self.image_shape = (image_shape, image_shape)

        if isinstance(space, Conv2DSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.desired_space = Conv2DSpace(shape = self.image_shape,
                                            channels = 1,
                                            axes = ('c', 0, 1, 'b'))

        self.output_space = VectorSpace(self.dim)

        rng = self.mlp.rng


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

class LocalRectified(ConvRectifiedLinear):

    def set_input_space(self, space):
        """
        .. todo::

            WRITEME

        Notes
        -----
        This resets parameters!
        """

        self.input_space = space
        rng = self.mlp.rng

        if not isinstance(self.input_space, Conv2DSpace):
            raise TypeError("The input to ths layer should be a Conv2DSpace"
                    " but layer " + self.layer_name + " got " + str(type(self.input_space)))

        if self.border_mode == 'valid':
            output_shape = [(self.input_space.shape[0] - self.kernel_shape[0]) / self.kernel_stride[0] + 1,
                (self.input_space.shape[1] - self.kernel_shape[1]) / self.kernel_stride[1] + 1]
        elif self.border_mode == 'full':
            output_shape = [(self.input_space.shape[0] +  self.kernel_shape[0]) / self.kernel_stride[0] - 1,
                    (self.input_space.shape[1] + self.kernel_shape[1]) / self.kernel_stride_stride[1] - 1]

        self.detector_space = Conv2DSpace(shape=output_shape,
                num_channels = self.output_channels,
                axes = ('b', 'c', 0, 1))

        if self.irange is not None:
            assert self.sparse_init is None
            # TODO make local_c01b accept space instead of space params
            self.transformer = local_c01b.make_random_local(
                input_groups = 1,
                irange = self.irange,
                input_axes = self.input_space.axes,
                image_shape = self.input_space.shape,
                output_axes = self.detector_space.axes,
                input_channels = self.input_space.num_channels,
                output_channels = self.detector_space.num_channels,
                kernel_shape = self.kernel_shape,
                kernel_stride = self.kernel_stride,
                # TODO
                pad = 0,
                partial_sum = 1,
                rng = rng)


        elif self.sparse_init is not None:
            raise NotImplementedError()
            #self.transformer = conv2d.make_sparse_random_conv2D(
                    #num_nonzero = self.sparse_init,
                    #input_space = self.input_space,
                    #output_space = self.detector_space,
                    #kernel_shape = self.kernel_shape,
                    #batch_size = self.mlp.batch_size,
                    #subsample = self.kernel_stride,
                    #border_mode = self.border_mode,
                    #rng = rng)

        W, = self.transformer.get_params()
        W.name = 'W'

        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
        self.b.name = 'b'

        print 'Input shape: ', self.input_space.shape
        print 'Detector space: ', self.detector_space.shape


        assert self.pool_type in ['max', 'mean']

        dummy_batch_size = self.mlp.batch_size
        if dummy_batch_size is None:
            dummy_batch_size = 2
        dummy_detector = sharedX(self.detector_space.get_origin_batch(dummy_batch_size))
        if self.pool_type == 'max':
            dummy_p = max_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        elif self.pool_type == 'mean':
            dummy_p = mean_pool(bc01=dummy_detector, pool_shape=self.pool_shape,
                    pool_stride=self.pool_stride,
                    image_shape=self.detector_space.shape)
        dummy_p = dummy_p.eval()
        self.output_space = Conv2DSpace(shape=[dummy_p.shape[2], dummy_p.shape[3]],
                num_channels = self.output_channels, axes = ('b', 'c', 0, 1) )

        print 'Output space: ', self.output_space.shape


    def get_filter_norms(self, W = None):

        if W is None:
            W, = self.transformer.get_params()

        assert W.ndim == 7
        sq_W = T.sqr(W)
        norms = T.sqrt(sq_W.sum(axis=(2, 3, 4)))
        return norms

    def get_monitoring_channels(self):
        """
        .. todo::

            WRITEME
        """

        filter_norms = self.get_filter_norms()
        return OrderedDict([
                            ('filter_norms_min'  , filter_norms.min()),
                            ('filter_norms_mean' , filter_norms.mean()),
                            ('filter_norms_max'  , filter_norms.max()),
                            ])

    def censor_updates(self, updates):
        """
        .. todo::

            WRITEME
        """

        if self.max_kernel_norm is not None:
            W ,= self.transformer.get_params()
            if W in updates:
                updated_W = updates[W]
                updated_norms = self.get_filter_norms(updated_W)
                desired_norms = T.clip(updated_norms, 0, self.max_kernel_norm)
                updates[W] = updated_W * (desired_norms / (1e-7 + updated_norms)
                        ).dimshuffle(0, 1, 'x', 'x', 'x', 2, 3)

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
    """
        This difference with Softmax is:
            * get the data in 1-D on convert to One_hot on the fly
            * return perplexity instead of missclass
    """
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


class MaxoutLocalC01BPoolLess(MaxoutLocalC01B):

     def fprop(self, state_below):
        """
        .. todo::

            WRITEME
        """

        self.input_space.validate(state_below)

        state_below = self.input_space.format_as(state_below, self.desired_space)

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
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s

            if self.detector_normalization:
                z = self.detector_normalization(z)

            if self.pool_shape is None:
                p = z
            else:
                raise NotImplementedError()
        else:

            if self.detector_normalization is not None:
                raise NotImplementedError("We can't normalize the detector "
                        "layer because the detector layer never exists as a "
                        "stage of processing in this implementation.")
            if self.pool_shape is not None:
                raise NotImplementedError()
            if self.num_pieces != 1:
                s = None
                for i in xrange(self.num_pieces):
                    t = z[i::self.num_pieces,:,:,:]
                    if s is None:
                        s = t
                    else:
                        s = T.maximum(s, t)
                z = s
            p = z


        self.output_space.validate(p)

        if hasattr(self, 'min_zero') and self.min_zero:
            p = p * (p > 0.)

        if not hasattr(self, 'output_normalization'):
            self.output_normalization = None

        if self.output_normalization:
            p = self.output_normalization(p)

        return p


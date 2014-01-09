import numpy as np
from pylearn2.models.mlp import Linear
from pylearn2.linear import local_c01b
from pylearn2.space import VectorSpace

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


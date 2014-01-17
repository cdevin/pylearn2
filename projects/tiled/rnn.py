from pylearn2.model.layer import Layer

class LinearRNN(Layer):


    def __init__(self,
                 dim,
                 layer_name,
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
        """
        Parameters
        ----------
        dim : WRITEME
        layer_name : WRITEME
        irange : WRITEME
        istdev : WRITEME
        sparse_init : WRITEME
        sparse_stdev : WRITEME
        include_prob : float
            Probability of including a weight element in the set of weights \
            initialized to U(-irange, irange). If not included it is \
            initialized to 0.
        init_bias : WRITEME
        W_lr_scale : WRITEME
        b_lr_scale : WRITEME
        mask_weights : WRITEME
        max_row_norm : WRITEME
        max_col_norm : WRITEME
        softmax_columns : WRITEME
        copy_input : WRITEME
        use_abs_loss : WRITEME
        use_bias : WRITEME
        """

        if use_bias and init_bias is None:
            init_bias = 0.

        self.__dict__.update(locals())
        del self.self

        if use_bias:
            self.b = sharedX( np.zeros((self.dim,)) + init_bias, name = layer_name + '_b')
        else:
            assert b_lr_scale is None
            init_bias is None



    @wraps(Layer.set_input_space)
    def set_input_space(self, space):

        self.input_space = space

        if isinstance(space, VectorSpace):
            self.requires_reformat = False
            self.input_dim = space.dim
        else:
            self.requires_reformat = True
            self.input_dim = space.get_total_dimension()
            self.desired_space = VectorSpace(self.input_dim)

        self.output_space = VectorSpace(self.dim + self.copy_input * self.input_dim)

        rng = self.mlp.rng
        if self.irange is not None:
            assert self.istdev is None
            assert self.sparse_init is None
            W_i = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0.,1., (self.input_dim, self.dim))
                 < self.include_prob)
            W_h = rng.uniform(-self.irange,
                            self.irange,
                            (self.input_dim, self.dim)) * \
                (rng.uniform(0.,1., (self.dim, self.dim))
                 < self.include_prob)
        elif self.istdev is not None:
            assert self.sparse_init is None
            W_i = rng.randn(self.input_dim, self.dim) * self.istdev
            W_h = rng.randn(self.dim, self.dim) * self.istdev
        else:
            raise NotImplementedError()

        W_i = sharedX(W_i)
        W_i.name = self.layer_name + '_W_input'

        self.input_transformer = MatrixMul(W_i)

        W_i ,= self.input_transformer.get_params()
        assert W_i.name is not None

        W_h = sharedX(W_h)
        W_h.name = self.layer_name + '_W_hidden'

        self.input_transformer = MatrixMul(W_h)

        W_h ,= self.input_transformer.get_params()
        assert W_h.name is not None

        if self.mask_weights is not None:
            raise NotImplementedError()


    @static_method
    def activation(x):
        return x

    def fprop(self, state_below):

        def recurrent(x, h):
            out = self.input_transformer(x) + \
                    self.hidden_transformer(x) + \
                    self.bias
            return self.activation(out)

        z, _ = theano.scan(recurrent,
                            sequenece = state_below,
                            outputs_info = [h0],
                            non_sequeces = self.params,
                            name = 'recurrent',
                            mode = theano.Mode(linker='cvm'))

        return z

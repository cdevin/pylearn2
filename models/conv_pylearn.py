import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.utils import sharedX
from pylearn2.space import Conv2DSpace, VectorSpace
#from pylearn2.linear.conv2d import Conv2D
from noisy_encoder.models.mlp_new import DropOutMLP
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal import downsample


class Conv2D:
    """ A temporarily solution for pylearn2.linear.conv2d.Conv2D
    slowness problem.
    TODO: It should be replaced by pylearn2 version after issue is resolved
    """
    def __init__(self,
                filters,
                batch_size,
                input_space,
                output_axes = ('b', 0, 1, 'c'),
                subsample = (1, 1),
                border_mode = 'valid',
                filters_shape = None,
                message = ""):

        self.filters = filters
        self.input_space = input_space
        self.output_axes = output_axes
        self.img_shape = (batch_size, input_space.nchannels,
                        input_space.shape[0], input_space.shape[1])
        self.subsample = subsample
        self.border_mode = border_mode
        self.filters_shape = filters.get_value(borrow=True).shape


    def lmul(self, x):

        assert x.ndim == 4
        axes = self.input_space.axes
        assert len(axes) == 4

        op_axes = ('b', 'c', 0, 1)

        if tuple(axes) != op_axes:
            x = x.dimshuffle(
                axes.index('b'),
                axes.index('c'),
                axes.index(0),
                axes.index(1))


        conv_out = conv2d(x, self.filters,
                        image_shape = self.img_shape,
                        filter_shape = self.filters_shape,
                        border_mode = self.border_mode)

        rval = downsample.max_pool_2d(input = conv_out,
                ds = self.subsample, ignore_border = True)

        axes = self.output_axes
        assert len(axes) == 4

        if tuple(axes) != op_axes:
            rval = rval.dimshuffle(
                    op_axes.index(axes[0]),
                    op_axes.index(axes[1]),
                    op_axes.index(axes[2]),
                    op_axes.index(axes[3]))
        return rval

class Conv(Block, Model):
    """ A convolution - pool block """

    def __init__(self,
                    image_shape,
                    kernel_shape,
                    nchannels_input,
                    nchannels_output,
                    pool_shape,
                    batch_size,
                    act_enc,
                    border_mode = 'valid',
                    irange = 0.05,
                    rng=9001):

        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState([2012,11,6,9])
        else:
            self.rng = rng

        self.nchannels_output = nchannels_output

        self.input_space = Conv2DSpace(shape = image_shape, nchannels = nchannels_input)
        self.output_space = Conv2DSpace(shape = [(a-b+1) / c for a, b, c in \
                zip(image_shape, kernel_shape, pool_shape)], nchannels = nchannels_output)

        self.weights = sharedX(self.rng.uniform(-irange,irange,(self.output_space.nchannels, self.input_space.nchannels, \
                      kernel_shape[0], kernel_shape[1])))
        self._initialize_hidbias()


        def _resolve_callable(conf, conf_attr):
            if conf[conf_attr] is None or conf[conf_attr] == "linear":
                return None
            if act_enc == "rectifier":
                return rectifier
            # If it's a callable, use it directly.
            if hasattr(conf[conf_attr], '__call__'):
                return conf[conf_attr]
            elif (conf[conf_attr] in globals()
                    and hasattr(globals()[conf[conf_attr]], '__call__')):
                return globals()[conf[conf_attr]]
            elif hasattr(tensor.nnet, conf[conf_attr]):
                return getattr(tensor.nnet, conf[conf_attr])
            elif hasattr(tensor, conf[conf_attr]):
                return getattr(tensor, conf[conf_attr])
            else:
                raise ValueError("Couldn't interpret %s value: '%s'" %
                        (conf_attr, conf[conf_attr]))


        self.act_enc = _resolve_callable(locals(), 'act_enc')

        # store parameters of this layer
        self._params = [self.weights, self.hidbias]

        self.transformer = Conv2D(filters = self.weights,
                batch_size = batch_size,
                input_space = self.input_space,
                output_axes = self.output_space.axes,
                subsample = pool_shape,
                border_mode = border_mode,
                filters_shape = self.weights.get_value().shape, message = "")


    def _initialize_hidbias(self):
        self.hidbias = sharedX(
            numpy.zeros(self.nchannels_output),
             name='hb',
            borrow=True
        )


    def _convolve(self, x):
        return self.transformer.lmul(x)

    def _hidden_input(self, x):
        return self._convolve(x) + self.hidbias

    def _hidden_activation(self, x):
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

    def encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def __call__(self, inputs):
        return self.encode(inputs)

    def get_weights(self):
        return self.weights

class LeNet(Block, Model):

    def __init__(self,
                    image_shape,
                    kernel_shapes,
                    nchannels,
                    pool_shapes,
                    batch_size,
                    conv_act,
                    mlp_act,
                    mlp_input_corruptors,
                    mlp_hidden_corruptors,
                    mlp_nunits,
                    n_outs,
                    border_mode = 'valid',
                    irange = 0.05,
                    rng=9001):


        self.layers = []
        self._params = []

        self.input_space = Conv2DSpace(shape = image_shape, nchannels = nchannels[0])
        self.output_space = VectorSpace(mlp_nunits[-1])

        for i in range(len(kernel_shapes)):
            layer = Conv(irange = irange,
                    image_shape = image_shape,
                    kernel_shape = kernel_shapes[i],
                    nchannels_input = nchannels[i],
                    nchannels_output = nchannels[i+1],
                    pool_shape = pool_shapes[i],
                    batch_size = batch_size,
                    act_enc = conv_act)
            self.layers.append(layer)
            self._params.extend(layer._params)

            image_shape = [(a - b + 1) / c for a, b, c in zip(image_shape,
                            kernel_shapes[i], pool_shapes[i])]

        mlp_nunits.insert(0, numpy.prod(image_shape) * nchannels[-1])
        self.mlp = DropOutMLP(input_corruptors = mlp_input_corruptors,
                        hidden_corruptors = mlp_hidden_corruptors,
                        n_units = mlp_nunits,
                        n_outs = n_outs,
                        act_enc = mlp_act)

        self._params.extend(self.mlp._params)

    def conv_encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return x.flatten(2)

    def encode(self, x):
        return self.mlp.encode(self.conv_encode(x))

    def test_encode(self, x):
        return self.mlp.test_encode(self.conv_encode(x))

    def p_y_given_x(self, inputs):
        return self.mlp.p_y_given_x(self.conv_encode(inputs))

    def predict_y(self, inputs):
        return self.mlp.predict_y(self.conv_encode(inputs))

    def __call__(self, inputs):
        return self.test_encode(inputs)

class LeNetLearner(object):
    "Temporary class to train LeNet with current non-pylearn sgd"

    def __init__(self,
                    image_shape,
                    kernel_shapes,
                    nchannels,
                    pool_shapes,
                    batch_size,
                    conv_act,
                    mlp_act,
                    mlp_input_corruptors,
                    mlp_hidden_corruptors,
                    mlp_nunits,
                    n_outs,
                    border_mode = 'valid',
                    irange = 0.05,
                    rng=9001):

        self.x = tensor.matrix('x')
        self.y = tensor.ivector('y')

        # This is the shape pylearn handles images batches
        self.input = self.x.reshape((batch_size, image_shape[0], image_shape[1], nchannels[0]))
        self.model = LeNet(image_shape = image_shape,
                    kernel_shapes = kernel_shapes,
                    nchannels = nchannels,
                    pool_shapes = pool_shapes,
                    batch_size = batch_size,
                    conv_act = conv_act,
                    mlp_act = mlp_act,
                    mlp_input_corruptors = mlp_input_corruptors,
                    mlp_hidden_corruptors = mlp_hidden_corruptors,
                    mlp_nunits = mlp_nunits,
                    n_outs = n_outs,
                    border_mode = border_mode,
                    irange = irange,
                    rng=rng)
        self.input_space = self.model.input_space
        self.params = self.model._params

    def errors(self, inputs, y):
        return tensor.mean(tensor.neq(self.model.predict_y(inputs), y))

    def negative_log_likelihood(self, inputs, y):
        return -tensor.mean(tensor.log(self.model.p_y_given_x(inputs))[tensor.arange(y.shape[0]), y])

    def encode(self, inputs):
        return self.model.encode(inputs)

    def test_encode(self, inputs):
        return self.model.test_encode(inputs)

    def __call__(self, inputs):
        return self.test_encode(inputs)

    def build_finetune_functions(self, datasets, batch_size, enable_momentum, w_l1_ratio = 0.0, act_l1_ratio = 0.0):

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        index = tensor.lscalar()  # index to a [mini]batch
        learning_rate = tensor.scalar('lr')
        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')
        cost = self.negative_log_likelihood(self.input, self.y)
        gparams = tensor.grad(cost, self.params)

        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self.params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self.params, gparams):
                inc = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc - learning_rate * gparam
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, cost],
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        errors = self.errors(self.input, self.y)

        test_score_i = theano.function([index], errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], errors,
              givens={
                 self.x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 self.y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score



import numpy
import theano
from theano import tensor
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.linear.conv2d import Conv2D
from pylearn2.utils import sharedX
from pylearn2.space import Conv2DSpace, VectorSpace
from pylearn2.corruption import GaussianCorruptor
from noisy_encoder.models.mlp import DropOutMLP
from noisy_encoder.models.dropouts import DeepDropOutHiddenLayer
from noisy_encoder.utils.corruptions import BinomialCorruptorScaled
from noisy_encoder.utils.normalize import LocalResponseNormalize
from noisy_encoder.models.base import rectifier


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
                    normalizers,
                    border_mode = 'valid',
                    irange = 0.05,
                    bias_init = 0.0,
                    rng=9001):

        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState([2012,11,6,9])
        else:
            self.rng = rng

        self.nchannels_output = nchannels_output
        self.pool_shape = pool_shape

        self.input_space = Conv2DSpace(shape = image_shape, nchannels = nchannels_input)
        self.output_space = Conv2DSpace(shape = [(a-b+1) / c for a, b, c in \
                zip(image_shape, kernel_shape, pool_shape)], nchannels = nchannels_output)

        self.weights = sharedX(self.rng.uniform(-irange,irange,(self.output_space.nchannels, self.input_space.nchannels, \
                      kernel_shape[0], kernel_shape[1])))
        self._initialize_hidbias(bias_init)


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

        assert isinstance(normalizers, list)
        self.normalizers = normalizers

        # store parameters of this layer
        self._params = [self.weights, self.hidbias]

        self.transformer = Conv2D(filters = self.weights,
                batch_size = batch_size,
                input_space = self.input_space,
                output_axes = self.output_space.axes,
                subsample = (1, 1),
                border_mode = border_mode,
                filters_shape = self.weights.get_value().shape, message = "")


    def _initialize_hidbias(self, bias_init):
        self.hidbias = sharedX(
            numpy.ones(self.nchannels_output) * bias_init,
             name='hb',
            borrow=True
        )


    def _convolve(self, x):
        return self.transformer.lmul(x)

    def _pool(self, x):
        axes = self.output_space.axes
        op_axes = ('b', 'c', 0, 1)
        x = Conv2DSpace.convert(x, axes, op_axes)
        x = downsample.max_pool_2d(input = x,
                ds = self.pool_shape,
                ignore_border = True)
        return Conv2DSpace.convert(x, op_axes, axes)

    def _hidden_input(self, x):
        return self._convolve(x) + self.hidbias

    def _hidden_activation(self, x):
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

    def _normalize(self, x):
        if len(self.normalizers) == 0:
            return x

        axes = self.output_space.axes
        op_axes = ('b', 'c', 0, 1)
        x = Conv2DSpace.convert(x, axes, op_axes)

        for normalizer in self.normalizers:
            if normalizer != None:
                x = normalizer(x)

        return Conv2DSpace.convert(x, op_axes, axes)

    def encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            inputs = self._hidden_activation(inputs)
            inputs = self._normalize(inputs)
            return self._pool(inputs)
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
                    normalizer = None,
                    border_mode = 'valid',
                    irange = 0.05,
                    bias_init = 0.0,
                    rng=9001):


        self.layers = []
        self._params = []

        self.input_space = Conv2DSpace(shape = image_shape, nchannels = nchannels[0])

        for i in range(len(kernel_shapes)):
            layer = Conv(
                    image_shape = image_shape,
                    kernel_shape = kernel_shapes[i],
                    nchannels_input = nchannels[i],
                    nchannels_output = nchannels[i+1],
                    pool_shape = pool_shapes[i],
                    batch_size = batch_size,
                    act_enc = conv_act,
                    normalizers = [normalizer[i]],
                    irange = irange,
                    bias_init = bias_init,
                    rng = rng)
            self.layers.append(layer)
            self._params.extend(layer._params)

            image_shape = [(a - b + 1) / c for a, b, c in zip(image_shape,
                            kernel_shapes[i], pool_shapes[i])]

        self.output_space = Conv2DSpace(shape = image_shape, nchannels = nchannels[-1])


    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, inputs):
        return self.encode(inputs)

class LeNetLearner(object):
    "Temporary class to train LeNet with current non-pylearn sgd"

    def __init__(self,
                    image_shape,
                    kernel_shapes,
                    nchannels,
                    pool_shapes,
                    batch_size,
                    conv_act,
                    normalize_params,
                    mlp_act,
                    mlp_input_corruption_levels,
                    mlp_hidden_corruption_levels,
                    mlp_nunits,
                    n_outs,
                    border_mode = 'valid',
                    irange = 0.05,
                    bias_init = 0.0,
                    random_filters = False,
                    rng=9001):

        # This is the shape pylearn handles images batches
        self.image_topo = (batch_size, image_shape[0], image_shape[1], nchannels[0])

        # make corruptors:
        mlp_input_corruptors = []
        for item in mlp_input_corruption_levels:
            if item == None or item == 0.0:
                mlp_input_corruptors.extend([None])
            else:
                mlp_input_corruptors.extend([GaussianCorruptor(corruption_level = item)])

        mlp_hidden_corruptors = []
        for item in mlp_hidden_corruption_levels:
            if item == None or item == 0.0:
                mlp_hidden_corruptors.extend([None])
            else:
                mlp_hidden_corruptors.extend([BinomialCorruptorScaled(corruption_level = item)])

        # make normalizers
        if normalize_params is None:
            normalizers = None
        else:
            normalizers = []
            for i, param in enumerate(normalize_params):
                normalizers.extend([LocalResponseNormalize(batch_size, **param)])


        self.conv = LeNet(image_shape = image_shape,
                    kernel_shapes = kernel_shapes,
                    nchannels = nchannels,
                    pool_shapes = pool_shapes,
                    batch_size = batch_size,
                    conv_act = conv_act,
                    normalizer = normalizers,
                    border_mode = border_mode,
                    irange = irange,
                    bias_init = bias_init,
                    rng=rng)

        mlp_nunits.insert(0, numpy.prod(self.conv.output_space.shape) * self.conv.output_space.nchannels)
        self.mlp = DropOutMLP(input_corruptors = mlp_input_corruptors,
                        hidden_corruptors = mlp_hidden_corruptors,
                        n_units = mlp_nunits,
                        n_outs = n_outs,
                        act_enc = mlp_act,
                        rng = rng)



        self.input_space = self.conv.input_space
        if random_filters:
            self._params = self.mlp._params
        else:
            self._params = self.conv._params + self.mlp._params

        self.w_l1 = tensor.sum([abs(item.weights).sum() for item in \
                self.mlp.hiddens.layers]) + abs(self.mlp.log_layer.W).sum()
        self.w_l2 = tensor.sum([(item.weights ** 2).sum() for item in \
                self.mlp.hiddens.layers]) + (self.mlp.log_layer.W ** 2).sum()


    def conv_encode(self, x):
        x = x.reshape(self.image_topo)
        return self.conv(x).flatten(2)

    def errors(self, inputs, y):
        return tensor.mean(tensor.neq(self.mlp.predict_y(self.conv_encode(inputs)), y))

    def negative_log_likelihood(self, inputs, y):
        return -tensor.mean(tensor.log(self.mlp.p_y_given_x(self.conv_encode(inputs)))[tensor.arange(y.shape[0]), y])

    def encode(self, inputs):
        return self.mlp.encode(self.conv_encode(inputs))

    def test_encode(self, inputs):
        return self.mlp.test_encode(self.conv_encode(inputs))

    def __call__(self, inputs):
        return self.test_encode(inputs)

    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        index = tensor.lscalar()  # index to a [mini]batch
        x = tensor.matrix('x')
        y = tensor.ivector('y')
        learning_rate = tensor.scalar('lr')

        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')
        cost = self.negative_log_likelihood(x, y)
        cost += coeffs['w_l1'] * self.w_l1 + coeffs['w_l2'] * self.w_l2
        gparams = tensor.grad(cost, self._params)
        errors = self.errors(x, y)
        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self._params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self._params, gparams):
                inc = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc - learning_rate * gparam
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, errors],
              updates=updates,
              givens={
                x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})


        test_score_i = theano.function([index], errors,
                 givens={
                   x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], errors,
              givens={
                 x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score

class LeNetLearnerMultiCategory(object):
    "Temporary class to train LeNet with current non-pylearn sgd"

    def __init__(self,
                    image_shape,
                    kernel_shapes,
                    nchannels,
                    pool_shapes,
                    batch_size,
                    conv_act,
                    normalize_params,
                    mlp_act,
                    mlp_input_corruption_levels,
                    mlp_hidden_corruption_levels,
                    mlp_nunits,
                    n_outs,
                    border_mode = 'valid',
                    irange = 0.05,
                    bias_init = 0.0,
                    rng=9001):

        self._params = []
        # This is the shape pylearn handles images batches
        self.image_topo = (batch_size, image_shape[0], image_shape[1], nchannels[0])

        # make corruptors:
        mlp_input_corruptors = []
        for item in mlp_input_corruption_levels:
            if item == None or item == 0.0:
                mlp_input_corruptors.extend([None])
            else:
                mlp_input_corruptors.extend([GaussianCorruptor(corruption_level = item)])

        mlp_hidden_corruptors = []
        for item in mlp_hidden_corruption_levels:
            if item == None or item == 0.0:
                mlp_hidden_corruptors.extend([None])
            else:
                mlp_hidden_corruptors.extend([BinomialCorruptorScaled(corruption_level = item)])

        # make normalizers
        if normalize_params is None:
            normalizers = None
        else:
            normalizers = []
            for i, param in enumerate(normalize_params):
                normalizers.extend([LocalResponseNormalize(batch_size, **param)])


        self.conv = LeNet(image_shape = image_shape,
                    kernel_shapes = kernel_shapes,
                    nchannels = nchannels,
                    pool_shapes = pool_shapes,
                    batch_size = batch_size,
                    conv_act = conv_act,
                    normalizer = normalizers,
                    border_mode = border_mode,
                    irange = irange,
                    bias_init = bias_init,
                    rng=rng)
        self.input_space = self.conv.input_space
        self._params.extend(self.conv._params)

        mlp_nunits.insert(0, numpy.prod(self.conv.output_space.shape) * self.conv.output_space.nchannels)
        self.hiddens = DeepDropOutHiddenLayer(
                            input_corruptors = mlp_input_corruptors[:-1],
                            hidden_corruptors = mlp_hidden_corruptors[:-1],
                            n_units = mlp_nunits[:-1],
                            act_enc = mlp_act,
                            irange = irange,
                            bias_init = bias_init,
                            rng = rng)

        self.loglayer = DeepDropOutHiddenLayer(
                            input_corruptors = [None],
                            hidden_corruptors = [None],
                            n_units = mlp_nunits[-2:],
                            act_enc = "sigmoid",
                            irange = irange,
                            bias_init = bias_init,
                            rng = rng)



        self._params.extend(self.hiddens._params)
        self._params.extend(self.loglayer._params)

        self.w_l1 = tensor.sum([abs(item.weights).sum() for item in \
                self.hiddens.layers]) + abs(self.loglayer.layers[0].weights).sum()
        self.w_l2 = tensor.sum([(item.weights ** 2).sum() for item in \
                self.hiddens.layers]) + (self.loglayer.layers[0].weights ** 2).sum()


    def cross_entropy(self, x, y):
        h = self.p_y_given_x(x)
        h_ = tensor.switch(tensor.lt(h, 0.00000001), -10, tensor.log(h))
        h_1 = tensor.switch(tensor.lt(1-h, 0.00000001), -10, tensor.log(1-h))
        return - (y * h_ + (1-y)*h_1).sum(axis=1).mean()

    def conv_encode(self, x):
        x = x.reshape(self.image_topo)
        return self.conv(x).flatten(2)

    def encode(self, inputs):
        return self.hiddens.encode(self.conv_encode(inputs))

    def test_encode(self, inputs):
        return self.hiddens.test_encode(self.conv_encode(inputs))

    def p_y_given_x(self, inputs):
        return self.loglayer.encode(self.encode(inputs))

    def p_y_given_x_test(self, inputs):
        return self.loglayer.test_encode(self.test_encode(inputs))

    def y_pred(self, inputs):
        return tensor.argmax(self.p_y_given_x_test(inputs), axis=1)

    def errors(self, inputs, y):
        y_pred = self.y_pred(inputs)
        return tensor.mean(tensor.neq(y[tensor.arange(y.shape[0]), y_pred], tensor.ones_like(y_pred)))

    def __call__(self, inputs):
        return self.test_encode(inputs)

    def apply(self, inputs):
        x = tensor.matrix('x')
        return theano.function([x], self.p_y_given_x_test(inputs))

    def build_finetune_functions(self, datasets, batch_size, coeffs, enable_momentum):

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        index = tensor.lscalar()  # index to a [mini]batch
        x = tensor.matrix('x')
        y = tensor.matrix('y')

        learning_rate = tensor.scalar('lr')
        if enable_momentum is None:
            momentum = None
        else:
            momentum = tensor.scalar('momentum')
        cost = self.cross_entropy(x, y)
        cost += coeffs['w_l1'] * self.w_l1 + coeffs['w_l2'] * self.w_l2
        gparams = tensor.grad(cost, self._params)
        errors = self.errors(x, y)
        # compute list of fine-tuning updates
        updates = {}
        if momentum is None:
            for param, gparam in zip(self._params, gparams):
                updates[param] = param - gparam * learning_rate
        else:
            for param, gparam in zip(self._params, gparams):
                inc = sharedX(param.get_value() * 0.)
                updated_inc = momentum * inc - learning_rate * gparam
                updates[inc] = updated_inc
                updates[param] = param + updated_inc


        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate),
                theano.Param(momentum)],
              outputs=[cost, errors],
              updates=updates,
              givens={
                x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})


        test_score_i = theano.function([index], errors,
                 givens={
                   x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], errors,
              givens={
                 x: valid_set_x[index * batch_size:
                                     (index + 1) * batch_size],
                 y: valid_set_y[index * batch_size:
                                     (index + 1) * batch_size]})

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score


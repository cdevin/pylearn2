import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import GaussianCorruptor, BinomialCorruptorScaled
from noisy_encoder.models.naenc import NoisyAutoencoder
from noisy_encoder.models.mlp import LogisticRegression, rectifier, PickableLambda




class ConvPool(object):
    """Pool Layer of a convolutional network """

    def __init__(self, corruptor, filter_shape, image_shape, poolsize, act_enc, rng=9001):
        assert image_shape[1] == filter_shape[1]

        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState(rng)
        else:
            self.rng = rng

        self.corruptor = corruptor
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.act_enc = act_enc

        fan_in = numpy.prod(filter_shape[1:])
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

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
        self.params = [self.W, self.b]


    def _conv_out(self, x):
        return conv.conv2d(input=x, filters=self.W,
                filter_shape=self.filter_shape, image_shape=self.image_shape)

    def _pooled_out(self, x):
        return  downsample.max_pool_2d(input=self._conv_out(x),
                                            ds=self.poolsize, ignore_border=True)

    def _hidden_input(self, x):
        return self._pooled_out(x) + self.b.dimshuffle('x', 0, 'x', 'x')

    def _hidden_activation(self, x):
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

    def _corrupted_hidden_activation(self, x):
        hidden = sel._hidden_activation(x)
        return self.corruptor(hidden)

    def encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self._corrupted_hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def test_encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def __call__(self, inputs):
        return self.encode(inputs)

class Conv(object):
    def __init__(self, numpy_rng, n_units, corruption_levels, n_outs, act_enc, gaussian_avg):

        self.conv_layers = []
        self.params = []
        self.n_layers = len(n_units) - 1

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.x = T.matrix('x')
        self.y = T.ivector('y')

        # input corruption
        if corruption_levels[0] != 0:
            input_corruptor = GaussianCorruptor(stdev = corruption_levels[i], avg = gaussian_avg)
        else:
            input_corruptor = PickableLambda()

        output_clean = self.x
        output_corrupted = input_corruptor(self.x)
        self.L1 = 0
        for i in xrange(self.n_layers):
            corruptor = BinomialCorruptorScaled(corruption_level = corruption_levels[i + 1])
            conv_layer = ConvPool(hidden_corruptor,
                                        filter_shape = filter_shape[i],
                                        image_shape = image_shape[i],
                                        poolsize = poolsize[i]
                                        act_enc = act_enc)
            self.conv_layers.append(hidden_layer)
            self.params.extend([conv_layer.hidbias, hidden_layer.weights])
            output_clean = conv_layer.test_encode(output_clean)
            output_corrupted = conv_layer(output_corrupted)
            self.L1 += abs(conv_layer.weights).sum()


        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input_clean=output_clean,
                         input_corrupted = output_corrupted,
                         n_in=n_hid, n_out=n_outs)

        self.params.extend(self.logLayer.params)
        self.L1 += abs(self.logLayer.W).sum()
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

    def build_finetune_functions(self, datasets, batch_size, l1_ratio):

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost + l1_ratio * self.L1, self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate)],
              outputs=self.finetune_cost,
              updates=updates,
              givens={
                self.x: train_set_x[index * batch_size:
                                    (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size:
                                    (index + 1) * batch_size]})

        test_score_i = theano.function([index], self.errors,
                 givens={
                   self.x: test_set_x[index * batch_size:
                                      (index + 1) * batch_size],
                   self.y: test_set_y[index * batch_size:
                                      (index + 1) * batch_size]})

        valid_score_i = theano.function([index], self.errors,
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


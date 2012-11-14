import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import GaussianCorruptor, BinomialCorruptorScaled
from noisy_encoder.models.naenc import NoisyAutoencoder
from noisy_encoder.models.mlp import LogisticRegression, rectifier, PickableLambda




class ConvPool(Block, Model):
    """Pool Layer of a convolutional network """

    def __init__(self, corruptor, filter_shape, image_shape, poolsize, act_enc, rng=9001):
        assert image_shape[1] == filter_shape[1]

        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState([2012,11,6,9])
        else:
            self.rng = rng


        self.weights = sharedX(rng.uniform(-irange,irange,(output_space.nchannels, input_space.nchannels, \
                      kernel_shape[0], kernel_shape[1])))
        self._initialize_hidbias()

        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.act_enc = act_enc

        self.input_space = Conv2DSpace(shape = , nchannels = )
        #self.ouput_space =


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
        self.params = [self.weights, self.hidbias]


        self.transformer = Conv2D(filters = self.weights,
                batch_size = batch_size,
                input_space = input_space,
                output_axes = output_spaces.axes,
                subsample = subsample,
                border_mode = border_mode,
                filter_shape = self.weights.get_value.shape, message = "")


    def _initialize_hidbias(self):
        self.hidbias = sharedX(
            numpy.zeros(self.nhid),
             name='hb',
            borrow=True
        )


    def _convolve(self, x):
        return self.transformer.lmul(x)

    def _hidden_input(self, x):
        return self._convolve(x) + self.b

    def _hidden_activation(self, x):
        if self.act_enc is None:
            act_enc = lambda x: x
        else:
            act_enc = self.act_enc
        return act_enc(self._hidden_input(x))

     def encode(self, inputs):
        if isinstance(inputs, tensor.Variable):
            return self._corrupted_hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def __call__(self, inputs):
        return self.encode(inputs)



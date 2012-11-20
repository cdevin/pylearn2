import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.utils import sharedX
from pylearn2.space import Conv2DSpace
from pylearn2.linear.conv2d import Conv2D


class Conv(Block, Model):
    """Pool Layer of a convolutional network """

    def __init__(self,
                    irange,
                    image_shape,
                    kernel_shape,
                    nchannels_input,
                    nchannels_output,
                    pool_shape,
                    batch_size,
                    act_enc,
                    border_mode = 'valid',
                    rng=9001):

        if not hasattr(rng, 'randn'):
            self.rng = numpy.random.RandomState([2012,11,6,9])
        else:
            self.rng = rng

        self.nchannels_output = nchannels_output
        self.input_space = Conv2DSpace(shape = image_shape, nchannels = nchannels_input)
        self.output_space = Conv2DSpace(shape = [a-b+1 for a, b in zip(image_shape, kernel_shape)], nchannels = nchannels_output)

        self.weights = sharedX(self.rng.uniform(-irange,irange,(self.output_space.nchannels, self.input_space.nchannels, \
                      kernel_shape[0], kernel_shape[1])))
        self._initialize_hidbias()

        self.act_enc = act_enc



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

import numpy
from theano import tensor
from pylearn2.autoencoder import Autoencoder
from pylearn2.corruption import Corruptor, BinomialCorruptor
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from noisy_encoder.models.base import rectifier


class StochasticHiddenLayer(Autoencoder):

    def __init__ (self, input_corru
            nvis, nhid, act_enc,
            irange=1., bias_init = 1., rng=9001):

        """
        irange : flaot, optional
            The weights are initialized by normal distrubution, irange is the variane
        """

        self.bias_init = bias_init

        super(DropOutHiddenLayer, self).__init__(
        nvis = nvis,
        nhid = nhid,
        act_enc = act_enc,
        act_dec = None,
        tied_weights = True,
        irange = irange,
        rng = rng)

        self._params = [self.hidbias, self.weights]

    def _initialize_weights(self, nvis, rng=None, irange=None):
        if rng is None:
            rng = self.rng
        if irange is None:
            irange = self.irange
        # TODO: use weight scaling factor if provided, Xavier's default else
        self.weights = sharedX(
            rng.normal(loc = 0.0, scale = irange, size = (nvis, self.nhid)),
            name='W',
            borrow=True
        )

    def _initialize_hidbias(self):
        self.hidbias = sharedX(
            numpy.ones(self.nhid) * self.bias_init,
            name='hb',
            borrow=True
        )

    def _hidden_activation(self, x):

        hidden = super(DropOutHiddenLayer, self)._hidden_activation(corrupted_x)
        return self.s_rng.binomial(p = hidden)

    def test_encode(self, inputs):

        if isinstance(inputs, tensor.Variable):
            return super(DropOutHiddenLayer, self)._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

class DeepDropOutHiddenLayer(Autoencoder):

    def __init__(self, input_corruptors,
                    hidden_corruptors,
                    n_units,
                    act_enc,
                    irange = 1e-3,
                    bias_init =  0.0,
                    rng = 9001):


        self.input_space = VectorSpace(n_units[0])
        self.output_space = VectorSpace(n_units[-1])

        self.layers = []
        self._params = []
        self.weights = []
        self.n_layers = len(n_units) - 1
        for i in range(self.n_layers):
            self.layers.append(DropOutHiddenLayer(input_corruptor = input_corruptors[i],
                                hidden_corruptor = hidden_corruptors[i],
                                nvis = n_units[i],
                                nhid = n_units[i+1],
                                act_enc = act_enc,
                                irange = irange,
                                bias_init = bias_init,
                                rng = rng))
            self._params.extend(self.layers[-1]._params)
            self.weights.append(self.layers[-1].weights)

    def encode(self, inputs):
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs

    def test_encode(self, inputs):

        outputs = inputs
        for layer in self.layers:
            outputs = layer.test_encode(outputs)
        return outputs


import numpy
from theano import tensor
from pylearn2.autoencoder import Autoencoder
from pylearn2.corruption import Corruptor, BinomialCorruptor
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from noisy_encoder.models.base import rectifier


class DropOutAutoencoder(Autoencoder):

    def __init__(self, input_corruptor, hidden_corruptor,
            nvis, nhid, act_enc, act_dec,
            tied_weights = False, irange=1e-3, rng=9001):

        # TODO This is hack, find the proper way to make rectifier accessible everywhere
        if act_enc == "rectifier":
            act_enc = rectifier
        if act_dec == "rectifier":
            act_dec = rectifier

        super(DropOutAutoencoder, self).__init__(
        nvis,
        nhid,
        act_enc,
        act_dec,
        tied_weights,
        irange,
        rng)

        self.input_corruptor = input_corruptor
        self.hidden_corruptor = hidden_corruptor

    def _hidden_activation(self, x):

        corrupted_x = self.input_corruptor(x)
        hidden = super(DropOutAutoencoder, self)._hidden_activation(corrupted_x)
        return self.hidden_corruptor(hidden)

    def test_encode(self, inputs):

        if isinstance(inputs, tensor.Variable):
            return super(DropOutAutoencoder, self)._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def test_reconstruct(self, inputs):
        raise NotImplementedError("Is there such a thing?")

class DropOutHiddenLayer(Autoencoder):

    def __init__(self, input_corruptor,
            hidden_corruptor,
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

        self.input_corruptor = input_corruptor
        self.hidden_corruptor = hidden_corruptor
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

        if self.input_corruptor is None:
            corrupted_x = x
        else:
            corrupted_x = self.input_corruptor(x)
        hidden = super(DropOutHiddenLayer, self)._hidden_activation(corrupted_x)
        if self.hidden_corruptor is None:
            return hidden
        else:
            return self.hidden_corruptor(hidden)

    def test_encode(self, inputs):

        if isinstance(inputs, tensor.Variable):
            return super(DropOutHiddenLayer, self)._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

    def _activation_grad(self, inputs):

        acts = self._hidden_input(inputs)
        hiddens = self.act_enc(acts)
        act_grad = tensor.grad(hiddens.sum(), acts)
        return act_grad

    def jacobian_h_x(self, inputs):

        act_grad = self._activation_grad(inputs)
        jacobian = self.weights * act_grad.dimshuffle(0, 'x', 1)
        return jacobian


class BalancedDropOutHiddenLayer(Autoencoder):

    def __init__(self, gaussian_corruptor,
            nvis, nhid, act_enc,
            irange=1e-3, rng=9001):

        super(BalancedDropOutHiddenLayer, self).__init__(
        nvis = nvis,
        nhid = nhid,
        act_enc = act_enc,
        act_dec = None,
        tied_weights = True,
        irange = irange,
        rng = rng)

        self.gaussian_corruptor = gaussian_corruptor
        self._params = [self.hidbias, self.weights]

    def _jjt(self, x):

        hidden_input = self._hidden_input(x)
        hidden = self.act_enc(hidden_input)
        act_grad = tensor.grad(hidden.sum(), hidden_input)
        jjt = tensor.dot(self.weights.T, self.weights) * (act_grad.dimshuffle(0, 'x', 1) ** 2.)
        return jjt.mean(axis = 1)

    def _binomial_corruptor(self,x):

        return BinomialCorruptor(corruption_level = self._jjt(x))

    def _hidden_activation(self, x):

        hidden = super(BalancedDropOutHiddenLayer, self)._hidden_activation(x)
        if self.gaussian_corruptor is not None:
            hidden = self.gaussian_corruptor(hidden)

        binomial_corruptor = self._binomial_corruptor(x)
        return binomial_corruptor(hidden)


    def test_encode(self, inputs):

        if isinstance(inputs, tensor.Variable):
            return super(BalancedDropOutHiddenLayer, self)._hidden_activation(inputs)
        else:
            return [self.encode(v) for v in inputs]

class DeepDropOutAutoencoder(Autoencoder):

    def __init__(self, input_corruptors,
                    hidden_corruptors,
                    n_units,
                    act_enc,
                    act_dec,
                    tied_weights = False,
                    irange = 1e-3, rng = 9001):

        self.input_space = VectorSpace(n_units[0])
        self.output_space = VectorSpace(n_units[-1])

        self.layers = []
        self._params = []
        self.weights = []
        self.n_layers = len(n_units) - 1
        for i in range(self.n_layers):
            self.layers.append(DropOutAutoencoder(input_corruptor = input_corruptors[i],
                                hidden_corruptor = hidden_corruptors[i],
                                nvis = n_units[i],
                                nhid = n_units[i+1],
                                act_enc = act_enc,
                                act_dec = act_dec,
                                tied_weights = tied_weights,
                                irange = irange,
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

    def reconstruct(self, inputs):

        hiddens = self.encode(inputs)
        for layer in reversed(self.layers):
            hiddens = layer.decode(hiddens)
        return hiddens

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


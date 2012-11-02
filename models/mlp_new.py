import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.space import VectorSpace
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import GaussianCorruptor, BinomialCorruptorScaled
from noisy_encoder.models.naenc import NoisyAutoencoder

def rectifier(X):
    return X * (X > 0.0)

class PickableLambda(object):

    def __call__(self, x):
        return x

class MLP(Block, Model):
    def __init__(self,  n_units, corruption_levels, n_outs, act_enc, gaussian_avg):

        self.layers = []
        self._params = []
        self.weights = []
        self.n_layers = len(n_units) - 1

        assert self.n_layers > 0


        if act_enc == "rectifier":
            act_enc = rectifier

        if corruption_levels[0] != 0:
            self.input_corruptor = GaussianCorruptor(stdev = corruption_levels[0], avg = gaussian_avg)
        else:
            self.input_corruptor = PickableLambda()

        for i in xrange(self.n_layers):
            hidden_corruptor = BinomialCorruptorScaled(corruption_level = corruption_levels[i + 1])
            layer = NoisyAutoencoder(None,
                                        hidden_corruptor,
                                        nvis = n_units[i],
                                        nhid = n_units[i+1],
                                        act_enc = act_enc,
                                        act_dec = None,
                                        tied_weights = True)
            self.layers.append(layer)
            self._params.extend([layer.hidbias, layer.weights])
            self.weights.extend([layer.weights])

        # We now need to add a logistic layer on top of the MLP
        self.log_layer = LogisticRegressionLayer(nvis = n_units[-1],
                                            nclasses = n_outs)
        self.weights.extend([self.log_layer.W])

        self._params.extend(self.log_layer._params)
        self.input_space = VectorSpace(n_units[0])
        self.output_space = VectorSpace(n_units[-1])

    def encode(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def test_encode(self, x):
        for layer in self.layers:
            x = layer.test_encode(x)
        return x

    def p_y_given_x(self, inp):
        return self.log_layer.p_y_given_x(self.encode(inp))

    def test_p_y_given_x(self, inp):
        return self.log_layer.p_y_given_x(self.test_encode(inp))

    def predict_y(self, inp):
        return tensor.argmax(self.test_p_y_given_x(inp), axis=1)

    def __call__(self, inp):
        return self.p_y_given_x(inp)

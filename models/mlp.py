import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.base import Block
from pylearn2.models import Model
from pylearn2.classifier import LogisticRegressionLayer
from pylearn2.space import VectorSpace
from noisy_encoder.models.dropouts import DeepDropOutHiddenLayer
from noisy_encoder.models.base import rectifier

class PickableLambda(object):

    def __call__(self, x):
        return x

class DropOutMLP(Block, Model):
    def __init__(self, input_corruptors, hidden_corruptors, n_units, n_outs, act_enc, irange = 0.05, bias_init = 0.0,  rng = 9001):


        if act_enc == "rectifier":
            act_enc = rectifier

        self._params = []
        self.hiddens = DeepDropOutHiddenLayer(input_corruptors = input_corruptors,
                                        hidden_corruptors = hidden_corruptors,
                                        n_units = n_units,
                                        act_enc = act_enc,
                                        irange = irange,
                                        bias_init = bias_init,
                                        rng = rng)

        self._params.extend(self.hiddens._params)
        self.weights = self.hiddens.weights

        # We now need to add a logistic layer on top of the MLP
        self.log_layer = LogisticRegressionLayer(nvis = n_units[-1],
                                            nclasses = n_outs)
        self.weights.extend([self.log_layer.W])
        self._params.extend(self.log_layer._params)
        self.input_space = VectorSpace(n_units[0])
        self.output_space = VectorSpace(n_units[-1])

    def encode(self, x):
        return self.hiddens(x)

    def test_encode(self, x):
        return self.hiddens.test_encode(x)

    def p_y_given_x(self, inp):
        return self.log_layer.p_y_given_x(self.encode(inp))

    def test_p_y_given_x(self, inp):
        return self.log_layer.p_y_given_x(self.test_encode(inp))

    def predict_y(self, inp):
        return tensor.argmax(self.test_p_y_given_x(inp), axis=1)

    def __call__(self, inp):
        return self.p_y_given_x(inp)

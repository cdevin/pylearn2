import pickle
import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.autoencoder import NoisyAutoEncoder
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError, MeanBinaryCrossEntropy
from pylearn2.corruption import BinomialNoise

class NAENC():

    def __init__(self, prob, nvis, nhid, act_enc, act_dec):

        self.inputs = tensor.matrix('input_x')
        self.act_enc = act_enc

        self.corruptor = BinomialNoise(corruption_level = prob)

        self.model = NoisyAutoEncoder(corruptor = self.corruptor,
                                nvis = nvis,
                                nhid = nhid,
                                act_enc = act_enc,
                                act_dec = act_dec)

    def cost_updates(self, learning_rate):
        """
        Return sgd cost and updates
        """

        # reconstruction cost
        cost = MeanBinaryCrossEntropy()(self.model, self.inputs)

        grads = tensor.grad(cost, self.model._params)
        updates = {}
        for param, gparam in zip(self.model._params, grads):
            updates[param] = param - gparam * learning_rate

        return updates, cost

    def train_funcs(self, data, batch_size):
        """
        return theano functions

        Returns:
            train_cost: SGD training function
            tain_ortho: Orthogonalizaion cost
            thr_fn: Find the thershold for selecting hidden unit values
        """

        index = tensor.lscalar('index')
        learning_rate = tensor.scalar('lr')

        cost_updates, cost = self.cost_updates(learning_rate)

        cost_fn = theano.function(inputs = [index,
                    theano.Param(learning_rate)],
                    outputs = cost,
                    updates = cost_updates,
                    givens = {self.inputs :\
                            data[index * batch_size : (index + 1) * batch_size]})

        return  cost_fn

    def features(self, data, batch_size):
        """
        Returns features/represenations
        """

        index = tensor.lscalar('index')

        return theano.function(inputs = [index],
                    outputs = self.model.encode(self.inputs),
                    givens = {self.inputs :\
                            data[index * batch_size : (index + 1) * batch_size]})

    def save_params(self, save_path, name):
        """ save model weights"""

        data = {}
        for item in self.model._params:
            data[item.name] = item.get_value()


        with open(save_path + name + '_params.pkl', 'wb') as outf:
            pickle.dump(data, outf, -1)

    def save_model(self, save_path, name):
        """ save the whole model """

        with open(save_path + name + '_model.pkl', 'wb') as outf:
            pickle.dump(self.model, outf, -1)



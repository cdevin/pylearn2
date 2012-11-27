import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import BinomialCorruptorScaledGroup, BinomialCorruptorScaled
from pylearn2.corruption import GaussianCorruptor
from pylearn2.utils import sharedX
from base import *

def BinomoalCorruptor(corruption_level, group_size = None):

    if group_sizes == None:
        return BinomialCorruptorScaled(corruption_level = corruption_level)
    else:
        return BinomialCorruptorScaledGroup(corruption_level = corruption_level, group_size = group_size)

class MLP(object):
    def __init__(self, numpy_rng, n_units, gaussian_corruption_levels,
                    binomial_corruption_levels, group_sizes, n_outs, act_enc,
                    irange, bias_init, group_corruption_levels = None, rng = 9001):

        self.hidden_layers = []
        self.sparsity = []
        self.params = []
        self.n_layers = len(n_units) - 1

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.rng = numpy.random.RandomState(rng)
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        if act_enc == "rectifier":
            act_enc = rectifier


        self.w_l1 = 0.
        self.act_l1 = 0.
        for i in xrange(self.n_layers):
            if i == 0:
                input_clean = self.x
                input_corrupted = GaussianCorruptor(stdev = gaussian_corruption_levels[i])(self.x)
                input_corrupted = BinomialCorruptor(corruption_level = binomial_corruption_levels[i], group_size = group_sizes[i])(input_corrupted)
            else:
                input_clean = self.hidden_layers[-1].output_clean
                input_corrupted = GaussianCorruptor(stdev = gaussian_corruption_levels[i])(self.hidden_layers[-1].output_corrupted)
                input_corrupted = BinomialCorruptorScaled(corruption_level = binomial_corruption_levels[i], group_size = group_sizes[i])(input_corrupted)

            hidden_layer = HiddenLayer(input_clean = input_clean,
                                        input_corrupted = input_corrupted,
                                        n_in = n_units[i],
                                        n_out = n_units[i+1],
                                        activation = act_enc,
                                        irange = irange,
                                        bias_init = bias_init,
                                        rng = self.rng)
            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)
            self.w_l1 += abs(hidden_layer.W).sum()
            self.act_l1 += abs(hidden_layer.output_corrupted).sum()
            #self.sparsity.append(1. - (T.gt(hidden_layer.output_clean, 0.0).sum(1).mean() / float(n_units[i+1])))
            self.sparsity.append((hidden_layer.output_clean>0).mean())
        # Logistic layer
        input_clean = self.hidden_layers[-1].output_clean
        input_corrupted = GaussianCorruptor(stdev = gaussian_corruption_levels[-1])(self.hidden_layers[-1].output_corrupted)
        input_corrupted = BinomialCorruptorScaled(corruption_level = binomial_corruption_levels[-1], group_size = group_sizes[-1])(input_corrupted)
        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input_clean=input_clean,
                         input_corrupted = input_corrupted,
                         n_in=n_units[-1], n_out=n_outs,
                         irange = irange,
                         bias_init = bias_init,
                         rng = self.rng)


        self.params.extend(self.logLayer.params)
        #self.L1 += abs(self.logLayer.W).sum()
        self.cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)
        self.sparsity = T.stack(self.sparsity)

    def build_finetune_functions(self, datasets, batch_size, w_l1_ratio, act_l1_ratio, enable_momentum):

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
        if enable_momentum is None:
            momentum = None
        else:
            momentum = T.scalar('momentum')

        # compute the gradients with respect to the model parameters
        cost = self.cost + w_l1_ratio * self.w_l1 + act_l1_ratio * self.act_l1
        gparams = T.grad(cost, self.params)

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
              outputs=[cost, self.sparsity],
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


import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import BinomialCorruptorScaledGroupCombined, BinomialCorruptorScaled
from noisy_encoder.models.naenc import NoisyAutoencoder, DropOutHiddenLayer, BalancedDropOutHiddenLayer
from pylearn2.corruption import GaussianCorruptor
from pylearn2.utils import sharedX


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input_clean, input_corrupted, n_in, n_out):
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)
        self.p_y_given_x_clean = T.nnet.softmax(T.dot(input_clean, self.W) + self.b)
        self.p_y_given_x_corrupted = T.nnet.softmax(T.dot(input_corrupted, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x_clean, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x_corrupted)[T.arange(y.shape[0]), y])

    def errors(self, y):

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

class PickableLambda(object):
    def __call__(self, x):
        return x

def rectifier(X):
    return X * (X > 0.0)

class MLP(object):
    def __init__(self, numpy_rng, n_units, gaussian_corruption_levels,
                    binomial_corruption_levels, group_sizes, n_outs, act_enc,
                    irange, group_corruption_levels = None):

        self.hidden_layers = []
        self.params = []
        self.n_layers = len(n_units) - 1

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        if act_enc == "rectifier":
            act_enc = rectifier

        output_clean = self.x
        output_corrupted = GaussianCorruptor(stdev = gaussian_corruption_levels[0])(self.x)

        self.w_l1 = 0.
        self.act_l1 = 0.
        for i in xrange(self.n_layers):
            if group_corruption_levels is not None:
                binomial_corruptor = BinomialCorruptorScaledGroupCombined(
                        corruption_level_group = group_corruption_levels[i],
                        corruption_level_individual = binomial_corruption_levels[i],
                        group_size = group_sizes[i])
            else:
                binomial_corruptor = BinomialCorruptorScaled(
                    corruption_level = binomial_corruption_levels[i])
            gaussian_corruptor = GaussianCorruptor(
                    stdev = gaussian_corruption_levels[i+1])

            hidden_layer = DropOutHiddenLayer([gaussian_corruptor, binomial_corruptor],
                                        nvis = n_units[i],
                                        nhid = n_units[i+1],
                                        act_enc = act_enc,
                                        irange = irange)
            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer._params)
            output_corrupted = hidden_layer(output_corrupted)
            output_clean = hidden_layer.test_encode(output_clean)
            self.w_l1 += abs(hidden_layer.weights).sum()
            self.act_l1 += abs(output_corrupted).sum()

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input_clean=output_clean,
                         input_corrupted = output_corrupted,
                         n_in=n_units[-1], n_out=n_outs)


        self.params.extend(self.logLayer.params)
        #self.L1 += abs(self.logLayer.W).sum()
        self.cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)

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
              outputs=cost,
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



class MLP_JJT(MLP):
    def __init__(self, numpy_rng, n_units, gaussian_corruption_levels,
                    binomial_corruption_levels, group_sizes, n_outs, act_enc,
                    irange, group_corruption_levels = None):

        self.hidden_layers = []
        self.params = []
        self.n_layers = len(n_units) - 1

        assert self.n_layers > 0

        theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        if act_enc == "rectifier":
            act_enc = rectifier

        output_clean = self.x
        output_corrupted = GaussianCorruptor(stdev = gaussian_corruption_levels[0])(self.x)

        self.w_l1 = 0.
        self.act_l1 = 0.
        for i in xrange(self.n_layers):
            if group_corruption_levels is not None:
                binomial_corruptor = BinomialCorruptorScaledGroupCombined(
                        corruption_level_group = group_corruption_levels[i],
                        corruption_level_individual = binomial_corruption_levels[i],
                        group_size = group_sizes[i])
            gaussian_corruptor = GaussianCorruptor(
                    stdev = gaussian_corruption_levels[i+1])

            hidden_layer = BalancedDropOutHiddenLayer(gaussian_corruptor,
                                        nvis = n_units[i],
                                        nhid = n_units[i+1],
                                        act_enc = act_enc,
                                        irange = irange)
            self.hidden_layers.append(hidden_layer)
            self.params.extend(hidden_layer._params)
            output_corrupted = hidden_layer(output_corrupted)
            output_clean = hidden_layer.test_encode(output_clean)
            self.w_l1 += abs(hidden_layer.weights).sum()
            self.act_l1 += abs(output_corrupted).sum()

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input_clean=output_clean,
                         input_corrupted = output_corrupted,
                         n_in=n_units[-1], n_out=n_outs)


        self.params.extend(self.logLayer.params)
        #self.L1 += abs(self.logLayer.W).sum()
        self.cost = self.logLayer.negative_log_likelihood(self.y)
        self.errors = self.logLayer.errors(self.y)



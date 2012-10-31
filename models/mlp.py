import numpy
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm
from noisy_encoder.utils.corruptions import GaussianCorruptor, BinomialCorruptorScaled
from noisy_encoder.models.naenc import NoisyAutoencoder


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
    def __init__(self, numpy_rng, n_units, corruption_levels, n_outs, act_enc, gaussian_avg):

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
        if corruption_levels[0] != 0:
            input_corruptor = GaussianCorruptor(stdev = corruption_levels[0], avg = gaussian_avg)
            output_corrupted = input_corruptor(self.x)
        else:
            output_corrupted = self.x
        self.L1 = 0
        for i in xrange(self.n_layers):
            hidden_corruptor = BinomialCorruptorScaled(corruption_level = corruption_levels[i + 1])

            hidden_layer = NoisyAutoencoder(None,
                                        hidden_corruptor,
                                        nvis = n_units[i],
                                        nhid = n_units[i+1],
                                        act_enc = act_enc,
                                        act_dec = None,
                                        tied_weights = True)
            self.hidden_layers.append(hidden_layer)
            self.params.extend([hidden_layer.hidbias, hidden_layer.weights])
            output_corrupted = hidden_layer(output_corrupted)
            output_clean = hidden_layer.test_encode(output_clean)
            self.L1 += abs(hidden_layer.weights).sum()

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
                         input_clean=output_clean,
                         input_corrupted = output_corrupted,
                         n_in=n_units[-1], n_out=n_outs)

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


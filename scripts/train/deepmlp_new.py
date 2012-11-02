import cPickle
import gzip
import os
import sys
import time

import numpy
import theano
import argparse
from theano import tensor
from pylearn2.utils import serial
from noisy_encoder.models.mlp_new import MLP
from sklearn.preprocessing import Scaler
from jobman.tools import DD
from utils.config import get_data_path, get_result_path
from DLN.utils.costs import WeightsL1Cost
from pylearn2.costs.supervised_cost import NegativeLogLikelihood

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from classify import shared_dataset, norm


RESULT_PATH = get_result_path()
DATA_PATH = get_data_path()

class DeepMLP(object):
    """ Very temp while not using pylearn training alg"""
    def __init__(self, numpy_rng, n_units, corruption_levels, n_outs, act_enc, gaussian_avg):

        self.x = T.matrix('x')
        self.y = T.ivector('y')
        mlp = MLP(n_units, corruption_levels, n_outs, act_enc, gaussian_avg)
        self.cost = NegativeLogLikelihood()(mlp, self.x, self.y)
        self.errors = T.neq(mlp.predict_y(self.x), self.y).mean()
        self.params = mlp._params
        self.model = mlp

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
        # TODO The L1 causes float64 error for some reason
        L1 = WeightsL1Cost(l1_ratio)(self.model, self.x)
        gparams = T.grad(self.cost , self.params)

        # compute list of fine-tuning updates
        updates = {}
        for param, gparam in zip(self.params, gparams):
            updates[param] = param - gparam * learning_rate

        train_fn = theano.function(inputs=[index,
                theano.Param(learning_rate)],
              outputs=self.cost,
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


def run_mlp(datasets, learning_rate_init, n_units, corruption_levels,
            n_outs, act_enc, l1_ratio, gaussian_avg, training_epochs,
            batch_size, lr_shrink_time , lr_dc_rate, save_frequency, save_name):

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    mlp = DeepMLP(numpy_rng, n_units, corruption_levels, n_outs, act_enc, gaussian_avg)

    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = mlp.build_finetune_functions(
                datasets=datasets, batch_size=batch_size,
                l1_ratio = l1_ratio)

    print '... finetunning the model'
    # early-stopping parameters
    patience = 1000 * n_train_batches  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):
            if epoch > lr_shrink_time:
                learning_rate = learning_rate_init / (1. + lr_dc_rate * epoch)
            else:
                learning_rate = learning_rate_init
            minibatch_avg_cost = train_fn(minibatch_index, learning_rate)
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if (this_validation_loss < best_validation_loss *
                        improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        epoch = epoch + 1
        if (epoch + 1) % save_frequency == 0:
            print "Saving the model"
            serial.save(save_name, mlp)

    print "Saving the model"
    serial.save(save_name, mlp)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return test_score * 100., best_validation_loss * 100.

def experiment(state, channel):
    print "Loading data..."
    train_set = serial.load(state.data_path + 'train.pkl')
    test_set = serial.load(state.data_path + 'test.pkl')
    if state.dataset == 'mnist':
        train_size = 50000
        nouts = 10
    elif state.dataset == 'cifar10':
        train_size = 40000
        nouts = 10
    elif state.dataset == 'cifar100':
        train_size = 40000
        nouts = 100
    else:
        raise NameError('Unknown dataset: {}').format(state.dataset)

    if state.scale:
        print "Scaling data..."
        scaler = Scaler()
        scaler.fit(train_set.X)
        train_set.X = scaler.transform(train_set.X)
        test_set.X = scaler.transform(test_set.X)

    if state.norm:
        print "Normalizing..."
        train_set.X = numpy.vstack([norm(x) for x in train_set.X])
        test_set.X = numpy.vstack([norm(x) for x in test_set.X])

    if state.shuffle:
        rng = numpy.random.RandomState(23027)
        rand_idx = rng.permutation(train_set.X.shape[0])
        train_x = train_set.X[rand_idx][:train_size]
        train_y = train_set.y[rand_idx][:train_size]
        valid_x = train_set.X[rand_idx][train_size:]
        valid_y = train_set.y[rand_idx][train_size:]
        train = shared_dataset(train_x, train_y)
        valid = shared_dataset(valid_x, valid_y)

        del train_set, train_x, train_y, valid_x, valid_y
    else:
        train = shared_dataset(train_set.X[:train_size], train_set.y[:train_size])
        valid = shared_dataset(train_set.X[train_size:-2], train_set.y[train_size:-2])

    test = shared_dataset(test_set.X, test_set.y)


    state.test_score, state.valid_score  = run_mlp((train, valid, test),
                    state.lr, state.n_units, state.corruption_levels, nouts,
                    state.act_enc, state.l1_ratio, state.gaussian_avg,
                    state.nepochs, state.batch_size, state.lr_shrink_time,
                    state.lr_dc_rate, state.save_frequency, state.save_name)

    return channel.COMPLETE

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'mlp classifer')
    parser.add_argument('-s', '--scale', action = "store_true", default = False, help = "scale data")
    parser.add_argument('-n', '--norm', action = "store_true", default = False, help = "normalize data")
    parser.add_argument('-l', '--lr', help = "learning rates/C list", default = 0.01, type = float)
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10', 'cifar100'], required = True)
    args = parser.parse_args()

    state = DD()
    #state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn3/")
    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/")
    state.scale = args.scale
    state.dataset = args.dataset
    state.norm = args.norm
    state.nepochs = 1000
    #state.act_enc = "sigmoid"
    state.act_enc = "rectifier"
    state.lr = args.lr
    state.lr_shrink_time = 50
    state.lr_dc_rate = 0.001
    state.batch_size = 50
    state.l1_ratio = 0.0
    state.gaussian_avg = 0.0
    state.shuffle = False
    state.n_units = [32*32*3, 1000, 10000]
    state.corruption_levels = [0.5, 0.5, 0.5]
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/cifar/mlp.pkl")

    experiment(state, None)


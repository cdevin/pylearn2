import time, sys, os
import numpy
import numpy as np
from copy import deepcopy
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.utils import serial
from noisy_encoder.training_algorithms.utils import LearningRateAdjuster, MomentumAdjuster
from pylearn2.utils.iteration import is_stochastic

def sgd_full(model,
            datasets,
            training_epochs,
            batch_size,
            coeffs,
            lr_params,
            save_frequency,
            save_name,
            enable_momentum,
            momentum_params):

    """
    Stochastic Gradient Decent
    on all data on all epochs
    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                datasets=datasets,
                batch_size=batch_size,
                coeffs=coeffs,
                enable_momentum = enable_momentum)

    print '... training the model'

    best_model = None
    best_epoch = 0
    best_train = numpy.inf
    best_valid = numpy.inf
    best_test = numpy.inf
    start_time = time.clock()


    monitors = {'cost': [],  'train' : [], 'valid' : [], 'test': []}
    lr_adjuster = LearningRateAdjuster(**lr_params)
    momentum_adjuster = MomentumAdjuster(**momentum_params)

    for epoch in xrange(training_epochs):
        learning_rate = lr_adjuster.get_value(epoch)
        momentum = momentum_adjuster.get_value(epoch)

        cost = []
        train_error = []
        for minibatch_index in xrange(n_train_batches):
            b_cost, b_train_error = train_fn(minibatch_index, learning_rate, momentum)
            if numpy.isnan(b_cost):
                print "NaN values showed up, have to stop here."
                break
            cost.append(b_cost)
            train_error.append(b_train_error)

        cost = numpy.mean(cost)
        train_error = numpy.mean(train_error)
        valid_error = numpy.mean(validate_model())
        test_error = numpy.mean(test_model())
        print "epoch {}, cost: {}, train: {}, valid: {}, test:{}".format(epoch, cost, train_error, valid_error, test_error)

        #save monitors
        monitors['cost'].append(cost)
        monitors['train'].append(train_error)
        monitors['valid'].append(valid_error)
        monitors['test'].append(test_error)

        # if we got the best validation score until now
        if valid_error < best_valid:
            best_valid = valid_error
            best_test = test_error
            best_train = train_error
            best_epoch = epoch
            best_model = deepcopy(model)
            print "\tBest one so far!"

        if (epoch + 1) % save_frequency == 0:
            print "Saving the model"
            serial.save(save_name, best_model)
            serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)
    end_time = time.clock()

    print "SGD finieshed with best validation error at epoch {} with:\n\ttrain error: {}\n\tvalid error: {}\n\ttest error: {}".format(
                    best_epoch, best_train, best_valid, best_test)
    print "The code took {}".format((end_time - start_time) / 60.)
    return best_test * 100., best_valid * 100.

def sgd(model,
            datasets,
            training_epochs,
            batch_size,
            coeffs,
            lr_params,
            save_frequency,
            save_name,
            enable_momentum,
            momentum_params):

    """
    Early stopping Stochastic Gradient Decent
    """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                datasets=datasets,
                batch_size=batch_size,
                coeffs=coeffs,
                enable_momentum = enable_momentum)

    print '... training the model'
    # early-stopping parameters
    patience = 10000 * n_train_batches  # look as this many examples regardless
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
    best_model = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    monitors = {'cost': [],  'train' : [], 'valid' : [], 'test': []}
    lr_adjuster = LearningRateAdjuster(**lr_params)
    momentum_adjuster = MomentumAdjuster(**momentum_params)

    while (epoch < training_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):
            learning_rate = lr_adjuster.get_value(epoch)
            momentum = momentum_adjuster.get_value(epoch)

            minibatch_avg_cost, train_score = train_fn(minibatch_index, learning_rate, momentum)
            iter = epoch * n_train_batches + minibatch_index
            if numpy.isnan(minibatch_avg_cost):
                done_looping = True
                break

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('%i, cost %f, lr %f, validation error %f %%' %
                      (epoch, minibatch_avg_cost, learning_rate, this_validation_loss * 100.))

                #save monitors

                monitors['cost'].append(minibatch_avg_cost)
                monitors['train'].append(train_score)
                monitors['valid'].append(this_validation_loss)
                monitors['test'].append(test_score)

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
                    try:
                        best_model = deepcopy(model)
                    except RuntimeError:
                        sys.setrecursionlimit(1500)
                        best_model = deepcopy(model)
                    print(('\t\t\t\t\tepoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break
        epoch = epoch + 1
        if (epoch + 1) % save_frequency == 0:
            print "Saving the model"
            serial.save(save_name, best_model)
            serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return test_score * 100., best_validation_loss * 100.

def sgd_mix(model,
            datasets,
            training_epochs,
            batch_size,
            coeffs,
            lr_params,
            save_frequency,
            save_name,
            enable_momentum,
            momentum_params):

    """
    This SGD train in turn on two set of mini-batches
    """

    train_set_x, train_set_y = datasets[0][0]

    # compute number of minibatches for training, validation and testing
    n_train_batches_0 = datasets[1][0][0].get_value(borrow=True).shape[0] / batch_size
    n_train_batches_1 = datasets[0][0][0].get_value(borrow=True).shape[0] / batch_size

    # get the training, validation and testing function for the model
    print '... getting the training functions'
    train_fn_0, train_fn_1, validate_model, test_model = model.build_finetune_functions(
                datasets=datasets,
                batch_size=batch_size,
                coeffs=coeffs,
                enable_momentum = enable_momentum)

    print '... training the model'
    # early-stopping parameters
    patience = 20
    patience_increase = 2
    improvement_threshold = 0.995
    validation_frequency = 1

    best_params = None
    best_model = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    monitors = {'cost_0': [], 'cost_1' :[], 'train_0' : [], 'train_1' : [], 'valid' : [], 'test': []}
    lr_adjuster_0 = LearningRateAdjuster(**lr_params[0])
    lr_adjuster_1 = LearningRateAdjuster(**lr_params[1])
    momentum_adjuster = MomentumAdjuster(**momentum_params)

    while (epoch < training_epochs) and (not done_looping):
        # Adjust learning rate
        learning_rate_0 = lr_adjuster_1.get_value(epoch)
        learning_rate_1 = lr_adjuster_0.get_value(epoch)
        momentum = momentum_adjuster.get_value(epoch)

        # first train
        cost_0 = []
        train_score_0 = []
        for minibatch_index in xrange(n_train_batches_0):
            cost, train_score = train_fn_0(minibatch_index, learning_rate_0, momentum)
            if numpy.isnan(cost):
                done_looping = True
                break
            cost_0.append(cost)
            train_score_0.append(train_score)
        cost_0 = numpy.mean(cost_0)
        train_score_0 = numpy.mean(train_score_0)

        # second train
        cost_1 = []
        train_score_1 = []
        for minibatch_index in xrange(n_train_batches_1):
            cost, train_score = train_fn_1(minibatch_index, learning_rate_1, momentum)
            if numpy.isnan(cost):
                done_looping = True
                break
            cost_1.append(cost)
            train_score_1.append(train_score)
        cost_1 = numpy.mean(cost_1)
        train_score_1 = numpy.mean(train_score_1)

        if (epoch + 1) % validation_frequency == 0:
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('%i, cost_0 %f, cost_1 %f,  validation error %f %%' %
                    (epoch, cost_0, cost_1, this_validation_loss * 100.))

            #save monitors

            monitors['cost_0'].append(cost_0)
            monitors['cost_1'].append(cost_1)
            monitors['train_0'].append(train_score_0)
            monitors['train_1'].append(train_score_1)
            monitors['valid'].append(this_validation_loss)
            monitors['test'].append(test_score)

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss *
                    improvement_threshold):
                    patience = max(patience, epoch * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = epoch

                # test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                best_model = deepcopy(model)
                print(('\t\t\t\t\tepoch %i,  test error of '
                        'best model %f %%') %
                        (epoch, test_score * 100.))

        if patience <= epoch:
            done_looping = True
            break
        epoch = epoch + 1

        if (epoch + 1) % save_frequency == 0:
            print "Saving the model"
            serial.save(save_name, best_model)
            serial.save(save_name.rstrip('.pkl') + '_monitor.pkl', monitors)

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return test_score * 100., best_validation_loss * 100.

def sgd_large(model,
            datasets,
            training_epochs,
            batch_size,
            coeffs,
            lr_params,
            save_frequency,
            save_name,
            enable_momentum,
            momentum_params):

    train_set = datasets[0]
    train_set_x, train_set_y = train_set.init_shared()


    # get the training, validation and testing function for the model
    print '... getting the training functions'
    train_fn, validate_model, test_model = model.build_finetune_functions(
                datasets= [(train_set_x, train_set_y), datasets[1], datasets[2]],
                batch_size=batch_size,
                coeffs=coeffs,
                enable_momentum = enable_momentum)

    print '... training the model'
    # early-stopping parameters
    n_train_batches = train_set.current_size
    patience = 10000 * n_train_batches  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                            # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = 1

    best_params = None
    best_model = None
    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    monitors = {'cost': [],  'train' : [], 'valid' : [], 'test': []}
    lr_adjuster = LearningRateAdjuster(**lr_params)
    momentum_adjuster = MomentumAdjuster(**momentum_params)
    while (epoch < training_epochs) and (not done_looping):
        # loop over shreded datasets
        for train_size in train_set:
            n_train_batches = train_size / batch_size
            for minibatch_index in xrange(n_train_batches):
                learning_rate = lr_adjuster.get_value(epoch)
                momentum = momentum_adjuster.get_value(epoch)

                minibatch_avg_cost, train_score = train_fn(minibatch_index, learning_rate, momentum)
                if numpy.isnan(minibatch_avg_cost):
                    done_looping = True
                    break

        if (epoch + 1) % validation_frequency == 0:
            validation_losses = validate_model()
            this_validation_loss = numpy.mean(validation_losses)
            print('%i, cost %f, lr %f, validation error %f %%' %
                  (epoch, minibatch_avg_cost, learning_rate, this_validation_loss * 100.))

            #save monitors

            monitors['cost'].append(minibatch_avg_cost)
            monitors['train'].append(train_score)
            monitors['valid'].append(this_validation_loss)
            monitors['test'].append(test_score)

            # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:

                #improve patience if loss improvement is good enough
                if (this_validation_loss < best_validation_loss *
                    improvement_threshold):
                    patience = max(patience, epoch * patience_increase)

                # save best validation score and iteration number
                best_validation_loss = this_validation_loss
                best_iter = epoch

                # test it on the test set
                test_losses = test_model()
                test_score = numpy.mean(test_losses)
                best_model = deepcopy(model)
                print(('\t\t\t\t\tepoch %i, minibatch %i/%i, test error of '
                       'best model %f %%') %
                      (epoch, minibatch_index + 1, n_train_batches,
                       test_score * 100.))

        if patience <= epoch:
            done_looping = True
            break
        epoch = epoch + 1
        if (epoch + 1) % save_frequency == 0:
            print "Saving the model"
            serial.save(save_name, best_model)
            serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save(save_name.rstrip('pkl') + '_monitor.pkl', monitors)
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return test_score * 100., best_validation_loss * 100.


class CrawlSGD(SGD):

    #def __init__(self, learning_rate, cost=None, batch_size=None,
                 #monitoring_batches=None, monitoring_dataset=None,
                 ##termination_criterion=None, update_callbacks=None,
                 ##init_momentum = None, set_batch_size = False,
                 ##train_iteration_mode = None, batches_per_iter=None,
                 ##theano_function_mode = None, monitoring_costs=None):


        ##super(CrawlSGD, self).__init__(self, learning_rate, cost=None, batch_size=None,
                 ##monitoring_batches=None, monitoring_dataset=None,
                 ##termination_criterion=None, update_callbacks=None,
                 ##init_momentum = None, set_batch_size = False,
                 ##train_iteration_mode = None, batches_per_iter=None,
                 ##theano_function_mode = None, monitoring_costs=None):


        ##self.terminate = False

    def train(self, dataset):
        if not hasattr(self, 'sgd_update'):
            raise Exception("train called without first calling setup")
        model = self.model
        batch_size = self.batch_size

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.first = False
        rng = self.rng
        if not is_stochastic(self.train_iteration_mode):
            rng = None
        if self.topo:
            batch_idx = dataset.get_topo_batch_axis()
        else:
            batch_idx = 0
        iterator = dataset.iterator(mode=self.train_iteration_mode,
                batch_size=self.batch_size, targets=self.supervised,
                topo=self.topo, rng = rng, num_batches = self.batches_per_iter,
                epoch = self.monitor.get_epochs_seen())
        if self.supervised:
            for (batch_in, batch_target) in iterator:
                self.sgd_update(batch_in, batch_target)
                actual_batch_size = batch_in.shape[batch_idx]
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)
        else:
            for batch in iterator:
                self.sgd_update(batch)
                actual_batch_size = batch.shape[0] # iterator might return a smaller batch if dataset size
                                                   # isn't divisible by batch_size
                self.monitor.report_batch(actual_batch_size)
                for callback in self.update_callbacks:
                    callback(self)

        # Make sure none of the parameters have bad values
        for param in self.params:
            value = param.get_value(borrow=True)
            if np.any(np.isnan(value)) or np.any(np.isinf(value)):
                raise Exception("NaN in " + param.name)

        self.terminate = True

    #def continue_learning(self, model):
        #if self.terminate:
            #return False
        #else:
            #return True

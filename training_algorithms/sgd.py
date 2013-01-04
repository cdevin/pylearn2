import time, sys, os
import numpy
from copy import deepcopy
from pylearn2.utils import serial


def sgd(model,
            datasets,
            learning_rate_init,
            training_epochs,
            batch_size,
            coeffs,
            lr_shrink_time,
            lr_dc_rate,
            save_frequency,
            save_name,
            enable_momentum,
            init_momentum,
            final_momentum,
            momentum_inc_start,
            momentum_inc_end):

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

    while (epoch < training_epochs) and (not done_looping):
        for minibatch_index in xrange(n_train_batches):
            # Adjust learning rate
            if epoch > lr_shrink_time:
                learning_rate = learning_rate_init / (1. + lr_dc_rate * epoch)
            else:
                learning_rate = learning_rate_init
            # Adjust Momentum
            if epoch < momentum_inc_start:
                momentum = init_momentum
            elif epoch < momentum_inc_end:
                momentum = init_momentum + ((final_momentum - init_momentum) / (momentum_inc_end - momentum_inc_start)) * (epoch - momentum_inc_start)
            else:
                momentum = final_momentum

            minibatch_avg_cost, train_score = train_fn(minibatch_index, learning_rate, momentum)
            iter = epoch * n_train_batches + minibatch_index
            if numpy.isnan(minibatch_avg_cost):
                done_looping = True
                break
            #if epoch == 2:
                #import ipdb
                #ipdb.set_trace()
            #minibatch_avg_cost, train_score = train_fn(minibatch_index, learning_rate, momentum)

            if (iter + 1) % validation_frequency == 0:
                #print model.mlp.hiddens.layers[0].weights.get_value().mean()
                #print model.mlp.log_layer.W.get_value().mean()
                #print model.mlp_p.layers[0].weights.get_value().mean()
                #print model.mlp_p.layers[1].weights.get_value().mean()
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
            serial.save('monitor.pkl', save_name.rstrip('pkl') + 'monitor.pkl')

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save('monitor.pkl', save_name.rstrip('pkl') + 'monitor.pkl')
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
            learning_rate_init,
            training_epochs,
            batch_size,
            coeffs,
            lr_shrink_time,
            lr_dc_rate,
            save_frequency,
            save_name,
            enable_momentum,
            init_momentum,
            final_momentum,
            momentum_inc_start,
            momentum_inc_end):

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

    while (epoch < training_epochs) and (not done_looping):
        # Adjust learning rate
        if epoch > lr_shrink_time:
            learning_rate_0 = learning_rate_init[0] / (1. + lr_dc_rate * epoch)
            learning_rate_1 = learning_rate_init[1] / (1. + lr_dc_rate * epoch)
        else:
            learning_rate_0 = learning_rate_init[0]
            learning_rate_1 = learning_rate_init[1]
        # Adjust Momentum
        if epoch < momentum_inc_start:
            momentum = init_momentum
        elif epoch < momentum_inc_end:
            momentum = init_momentum + ((final_momentum - init_momentum) / (momentum_inc_end - momentum_inc_start)) * (epoch - momentum_inc_start)
        else:
            momentum = final_momentum

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
            serial.save('monitor.pkl', save_name.rstrip('pkl') + 'monitor.pkl')

    print "Saving the model"
    serial.save(save_name, best_model)
    serial.save('monitor.pkl', save_name.rstrip('pkl') + 'monitor.pkl')
    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

    return test_score * 100., best_validation_loss * 100.

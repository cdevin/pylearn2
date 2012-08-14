import pickle, os, sys, time
import numpy
import theano
from theano import tensor
from util.funcs import load_tfd
from util.config import DATA_PATH



class LogisticRegression(object):


    def __init__(self, train_input, test_input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b')

        self.p_y_given_x_train = tensor.nnet.softmax(tensor.dot(train_input, self.W) + self.b)
        self.y_pred_train = tensor.argmax(self.p_y_given_x_train, axis=1)

        self.p_y_given_x_test = tensor.nnet.softmax(tensor.dot(test_input, self.W) + self.b)
        self.y_pred_test = tensor.argmax(self.p_y_given_x_test, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):

       return -tensor.mean(tensor.log(self.p_y_given_x_train)[tensor.arange(y.shape[0]), y])

    def errors(self, y):
        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred_test.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred_test.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tensor.mean(tensor.neq(self.y_pred_test, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    def __init__(self, rng, train_input, test_input, n_in, n_out,
                W=None, b=None, activation=tensor.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W')

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b

        lin_output_train = tensor.dot(train_input, self.W) + self.b
        self.train_output = (lin_output_train if activation is None
                       else activation(lin_output_train))

        lin_output_test = tensor.dot(test_input, self.W) + self.b
        self.test_output = (lin_output_test if activation is None
                       else activation(lin_output_test))

        # parameters of the model
        self.params = [self.W, self.b]

class FINETUNE():

    def __init__(self, models, n_hid, n_out):

        self.x = tensor.matrix('x')
        self.y = tensor.ivector('y')

        rng = numpy.random.RandomState(1234)

        self.layers = []
        self.params = []
        train_inputs = [self.x]
        test_inputs = [self.x]
        weights = []

        # noisy enoder layers
        for ind in range(len(models)):
            self.layers.append(models[ind])
            train_inputs.append(self.layers[-1](train_inputs[-1]))
            test_inputs.append(self.layers[-1].test_encode(test_inputs[-1]))
            self.params.extend([self.layers[-1].weights, self.layers[-1].hidbias])
            weights.append(self.layers[-1].weights)

        # hidden layer
        self.layers.append(HiddenLayer(rng = rng, train_input = train_inputs[-1],
                                        test_input = test_inputs[-1],
                                        n_in = self.layers[-1].nhid,
                                        n_out = n_hid))

        weights.append(self.layers[-1].W)

        # logistic layer
        self.loglayer = LogisticRegression(train_input = self.layers[-1].train_output,
                                            test_input = self.layers[-1].test_output,
                                            n_in = n_hid,
                                            n_out = n_out)

        weights.append(self.layers[-1].W)

        self.L1 = numpy.sum([abs(item).sum() for item in weights])
        self.L2 = numpy.sum([abs(item ** 2).sum() for item in weights])

        self.negative_log_likelihood = self.loglayer.negative_log_likelihood(self.y)
        self.error = self.loglayer.errors(self.y)
        self.params.extend(self.loglayer.params)

    def train_func(self, data_x, data_y, batch_size):

        index = tensor.lscalar('index')
        learning_rate = tensor.scalar('lr')

        cost = self.negative_log_likelihood + self.L1 + self.L2

        grads = tensor.grad(cost, self.params)
        updates = {}
        for param, gparam in zip(self.params, grads):
            updates[param] = param - gparam * learning_rate

        return theano.function(inputs = [index, theano.Param(learning_rate)],
                outputs = cost,
                updates = updates,
                givens={self.x: data_x[index * batch_size: (index + 1) * batch_size],
                        self.y: data_y[index * batch_size: (index + 1) * batch_size]})


    def test_func(self, data_x, data_y, batch_size):

        index = tensor.lscalar('index')

        return theano.function(inputs = [index],
                outputs = self.error,
                givens={self.x: data_x[index * batch_size: (index + 1) * batch_size],
                        self.y: data_y[index * batch_size: (index + 1) * batch_size]})

def load_models(paths):

    models = []
    for item in paths:
        model = pickle.load(open(item, 'r'))
        models.append(model)

    return models


def load_data(dataset, data_path, ds_type, fold, scale, scaler = None):

    if dataset == 'tfd':
        return load_tfd(data_path = data_path,
                        fold = fold,
                        ds_type = ds_type,
                        scale = scale,
                        scaler = scaler)
    if dataset == 'mnist':
        return load_mnist(data_path, ds_type = ds_type)
    else:
        raise NameError("Invalid dataset: {}".format(dataset))



def classify(model_paths,
            dataset,
            data_path,
            scale,
            fold,
            n_hid,
            n_out,
            lr,
            batch_size):


    network = FINETUNE(models = load_models(model_paths),
                        n_hid = n_hid,
                        n_out = n_out)


    train_x, train_y, scaler = load_data(dataset, data_path, 'train', fold, scale)
    valid_x, valid_y, _ = load_data(dataset, data_path, 'valid', fold, scale, scaler)
    test_x, test_y, _ = load_data(dataset, data_path, 'test', fold, scale, scaler)

    n_train_batches = train_x.get_value().shape[0] / batch_size
    n_valid_batches = valid_x.get_value().shape[0] / batch_size
    n_test_batches = test_x.get_value().shape[0] / batch_size

    train_model = network.train_func(train_x, train_y, batch_size)
    validate_model = network.test_func(valid_x, valid_y, batch_size)
    test_model = network.test_func(test_x, test_y, batch_size)

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
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
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index, learning_rate)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                     (epoch, minibatch_index + 1, n_train_batches,
                      this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    # test it on the test set

                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                    done_looping = True
                    break

    end_time = time.clock()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    valid_result, test_result = classify(model_paths = state.model_path,
            dataset = state.dataset,
            data_path = state.data_path,
            scale = state.scale,
            fold = state.fold,
            n_hid = state.nhid,
            n_out = state.nout,
            lr = state.lr,
            batch_size= state.batch_size)


    state.valid_result = valid_result
    state.test_result = test_result
    state.c_score = c_score

    return channel.COMPLETE

def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.dataset = "tfd"
    state.model_path = ["data/tfd_test_0_model.pkl"]
    state.data_path = DATA_PATH + "faces/TFD/raw/"
    state.scale = True
    state.fold = 0
    state.nhid = 1024
    state.nout = 7
    state.batch_size = 100
    state.lr = 0.1
    state.exp_name = 'test_run'

    experiment(state, None)

if __name__ == "__main__":

    test_experiment()

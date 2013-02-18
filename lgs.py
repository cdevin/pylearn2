import pickle, sys
import numpy
import theano
from theano import tensor
from time import time
from sklearn.svm import SVC
from util.funcs import load_tfd, load_mnist, features
from util.config import DATA_PATH

def load_data(dataset, data_path, ds_type, fold, scale):

    if dataset == 'tfd':
        return load_tfd(data_path = data_path,
                        fold = fold,
                        ds_type = ds_type,
                        scale = scale)
    if dataset == 'mnist':
        return load_mnist(data_path, ds_type = ds_type)
    else:
        raise NameError("Invalid dataset: {}".format(dataset))

def shared_dataset(data_x, data_y, shared = True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
        if shared == True:
            data_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
            data_y = tensor.cast(data_y, 'int32')

        return data_x, data_y


class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W')
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b')

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = tensor.nnet.softmax(tensor.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = tensor.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -tensor.mean(tensor.log(self.p_y_given_x)[tensor.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return tensor.mean(tensor.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

def classify_fns(data,
            n_in,
            batch_size):

    train_set_x, train_set_y = data[0]
    valid_set_x, valid_set_y = data[1]
    test_set_x, test_set_y = data[2]


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = tensor.lscalar()  # index to a [mini]batch
    learning_rate = tensor.scalar('lr')
    x = tensor.matrix('x')  # the data is presented as rasterized images
    y = tensor.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input= x, n_in= n_in, n_out=10)

    # the cost we minimize during training is the negative log likelihood of
    # the model in symbolic format
    cost = classifier.negative_log_likelihood(y)

    # compiling a Theano function that computes the mistakes that are made by
    # the model on a minibatch
    test_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function(inputs=[index],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]})

    # compute the gradient of cost with respect to theta = (W,b)
    g_W = tensor.grad(cost=cost, wrt=classifier.W)
    g_b = tensor.grad(cost=cost, wrt=classifier.b)

    # specify how to update the parameters of the model as a dictionary
    updates = {classifier.W: classifier.W - learning_rate * g_W,
               classifier.b: classifier.b - learning_rate * g_b}

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index,
            theano.Param(learning_rate)],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    return train_model, validate_model, test_model



def classify(model,
            dataset,
            data_path,
            scale,
            n_in,
            fold,
            lr_init,
            lr_decay,
            batch_size = 100,
            n_epochs = 500):


    # prepare data
    train_set_x, train_set_y = load_data(dataset, data_path, 'train', fold, scale)
    valid_set_x, valid_set_y = load_data(dataset, data_path, 'valid', fold, scale)
    test_set_x, test_set_y = load_data(dataset, data_path, 'test', fold, scale)

    train_set_x = features(model, train_set_x, batch_size)
    valid_set_x = features(model, valid_set_x, batch_size)
    test_set_x = features(model, test_set_x, batch_size)

    data = [shared_dataset(train_set_x, train_set_y),
            shared_dataset(valid_set_x, valid_set_y),
            shared_dataset(test_set_x, test_set_y)]

    # get functions
    train_model, validate_model, test_model = classify_fns(data, n_in, batch_size)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.shape[0] / batch_size
    n_valid_batches = valid_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] /batch_size



    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 500000  # look as this many examples regardless
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
    start_time = time()

    done_looping = False
    epoch = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            if lr_decay == -1:
                learning_rate = lr_init
            else:
                learning_rate = lr_init * lr_decay / (lr_decay + epoch)

            minibatch_avg_cost = train_model(minibatch_index, lr = learning_rate)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
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

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    #print 'The code run for %d epochs, with %f epochs/sec' % (
        #epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code ran for %.1fs' % ((end_time - start_time)))


    return best_validation_loss, test_score


def cross_validate(state):

    best_valid = numpy.inf
    best_test = numpy.inf
    for lr_init in state.lr_init:
        for lr_decay in state.lr_decay:
            valid, test = classify(model = state.model_path,
                dataset = state.dataset,
                data_path = state.data_path,
                scale = state.scale,
                n_in = state.nhid,
                fold = state.fold,
                batch_size= state.batch_size,
                lr_init = lr_init,
                lr_decay = lr_decay,
                n_epochs = state.n_epochs)

            if valid < best_valid:
                best_valid = valid
                best_test = test
    best_valid = 100 - best_valid*100
    best_test = 100- best_test*100

    print "Best valid result: {}, best test reslut: {}".format(best_valid, best_test)
    return best_valid, best_test


def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    valid_result, test_result  = cross_validate(state)

    state.valid_result = valid_result
    state.test_result = test_result

    return channel.COMPLETE

def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    #state.dataset = "tfd"
    state.dataset = "mnist"
    state.model_path = "data/mnist_12_model.pkl"
    state.data_path = DATA_PATH + "mnist/"
    state.scale = True
    state.nhid = 1024
    state.batch_size = 200
    state.n_epochs = 20
    state.c_vals = [3, 10, 10]
    state.fold = 0
    state.lr_decay = [-1, 1, 2, 5, 10]
    state.lr_init = [0.9, 0.5, 0.1, 0.05]
    state.exp_name = 'test_run'

    experiment(state, None)

if __name__ == "__main__":

    test_experiment()

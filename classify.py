import pickle, sys, os, fnmatch
import numpy
import theano
from theano import tensor
from io import load_tfd
from time import time



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


def classify(model,
            dataset,
            n_in,
            fold,
            learning_rate=0.13,
            n_epochs=1000,
            batch_size=600):


    train_set_x, train_set_y = load_data(dataset, 'train', fold)
    valid_set_x, valid_set_y = load_data(dataset, 'valid', fold)
    test_set_x, test_set_y = load_data(dataset, 'test', fold)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = tensor.lscalar()  # index to a [mini]batch
    x = tensor.matrix('x')  # the data is presented as rasterized images
    y = tensor.ivector('y')  # the labels are presented as 1D vector of
                           # [int] labels

    # construct the logistic regression class
    # Each MNIST image has size 28*28
    classifier = LogisticRegression(input= model.test_encode(x), n_in= n_in, n_out=7)

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
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size:(index + 1) * batch_size],
                y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
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

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = epoch * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                #print('epoch %i, minibatch %i/%i, validation error %f %%' % \
                        #(epoch, minibatch_index + 1, n_train_batches,
                    #this_validation_loss * 100.))

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

                    #print(('     epoch %i, minibatch %i/%i, test error of best'
                       #' model %f %%') %
                        #(epoch, minibatch_index + 1, n_train_batches,
                         #test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    #print 'The code run for %d epochs, with %f epochs/sec' % (
        #epoch, 1. * epoch / (end_time - start_time))
    #print >> sys.stderr, ('The code ran for %.1fs' % ((end_time - start_time)))


    return best_validation_loss, test_score


def load_model(data_path):

    return pickle.load(open(data_path, 'r'))


def load_data(dataset, ds_type, fold):

    if dataset == 'tfd':
        return load_tfd(fold, ds_type)
    else:
        raise NameError("Invalid dataset :{}".format(dataset))

def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    model = load_model(state.model_path)

    valid_result, test_result = classify(model = model,
            dataset = state.dataset,
            n_in = state.nhid,
            fold = state.fold,
            learning_rate= state.learning_rate,
            n_epochs= state.n_epochs,
            batch_size= state.batch_size)


    state.valid_result = valid_result
    state.test_result = test_result

    #return channel.COMPLETE


def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.data_path = "tfd"
    state.model_path = "jjt_test_3_model.pkl"
    state.nhid = 20
    state.learning_rate = 0.05
    state.batch_size = 50
    state.n_epochs = 1000
    state.fold = 0
    state.exp_name = 'test_run'

    experiment(state, None)

def batch_test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.dataset = "tfd"
    state.exp_path = "data/"
    state.nhid = 1024
    state.learning_rate = 0.05
    state.batch_size = 50
    state.n_epochs = 1000
    state.fold = 0
    state.exp_name = 'test_run'


    for root, dirnames, filenames in os.walk(state.exp_path):
        for filename in fnmatch.filter(filenames, '*_model.pkl'):
            state.model_path = os.path.join(root, filename)
            print "\n------\n running model {}\n".format(state.model_path)
            experiment(state, None)


if __name__ == "__main__":

    #test_experiment()
    batch_test_experiment()

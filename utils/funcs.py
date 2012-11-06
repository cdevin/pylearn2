import numpy
import pickle, gzip
import theano
from theano import tensor
from sklearn.preprocessing import Scaler

def scale_data(data, scaler = None):

    if scaler == None:
        scaler = Scaler().fit(data)

    return scaler.transform(data), scaler


def load_tfd(data_path, fold = 0, ds_type = 'train', scale = False, shared = False, scaler = None):

    # patches
    if type(fold) == type('str'):
        path = ["{}patches/{}.pkl".format(data_path, fold)]
    # unserpvised
    elif fold == -1:
        path = ["{}unsupervised/TFD_unsupervised_train_unlabeled_all.pkl".format(data_path)]
    # folds
    else:
        if ds_type not in ['train', 'valid', 'test']:
            raise NameError("wrong dataset type: {}".format(ds_type))
        if ds_type == "train":
            ds_type = 'train_labeled'

        path = ["{}FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(data_path, fold, fold, ds_type)]

    data_x, data_y = [], []
    for item in path:
        x, y = pickle.load(open(item, 'r'))
        data_x.append(x)
        data_y.append(y)

    data_x = numpy.concatenate(data_x).astype(theano.config.floatX)
    data_y = numpy.concatenate(data_y)

    def norm(X):
        s = X.std(0)
        m = X.mean(0)
        s = s + 0.0001*(s==0)
        return (X-m)


    if scale == True:
        data_x = data_x / 255.
        data_x = numpy.vstack([norm(x) for x in data_x])
        #data_x, scaler = scale_data(data_x, scaler)

    data_x = theano.shared(data_x)
    data_y = data_y.reshape(data_y.shape[0]) - 1
    if shared == True:
        data_y = theano.shared(data_y.astype(theano.config.floatX))
        data_y = theano.tensor.cast(data_y, 'int32')

    return data_x, data_y, scaler


def load_mnist(data_path, ds_type, shared = False, norm = False):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''
    print '... loading data'

    # Load the dataset
    f = gzip.open(data_path + "mnist.pkl.gz",'rb')
    train_set, valid_set, test_set = pickle.load(f)
    f.close()
    if ds_type == "train":
        data_x, data_y = train_set
    elif ds_type == "valid":
        data_x, data_y = valid_set
    elif ds_type == "test":
        data_x, data_y = test_set
    else:
        raise NamError("Invalid set type %s" %(set_type))

    def norm(X):
        s = X.std(0)
        m = X.mean(0)
        s = s + 0.0001*(s==0)
        return (X-m)/s

    def shared_dataset(data_x, data_y, shared):
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

    if norm == True:
        data_x = norm(data_x)


    data_x, data_y = shared_dataset(data_x, data_y, shared)

    return data_x, data_y



def features(model, data, batch_size = 100, scaler = None):

    def features_fn(model, data, batch_size):
        index = tensor.lscalar('index')
        x = tensor.matrix('x')
        return theano.function(inputs = [index],
                        outputs = model.test_encode(x),
                        givens = {x: data[index * batch_size : (index + 1) * batch_size]})


    model = pickle.load(open(model, 'r'))

    n_samples = data.get_value(borrow=True).shape[0]
    func = features_fn(model, data, batch_size)
    feats = [func(i) for i in xrange(n_samples / batch_size)]
    if numpy.mod(n_samples, batch_size) != 0:
        func = features_fn(model, data[(n_samples / batch_size) * batch_size : ],
                    numpy.mod(n_samples, batch_size))
        feats.append(func(0))

    def norm(X):
        s = X.std(0)
        m = X.mean(0)
        s = s + 0.0001*(s==0)
        return (X-m)/s


    res = numpy.concatenate(feats)
    res = numpy.vstack([norm(x) for x in res])
    return scale_data(res, scaler)


def learning_rate_adjuster(current_cost, previous_cost, lr, lr_change_tr):

    if current_cost > previous_cost:
        rval = lr * lr_change_tr
        print "Shrinking learning rate to: {}".format(rval)
    elif current_cost < previous_cost:
        rval = lr * (1 + lr_change_tr)
        print "Expanding learning rate to: {}".format(rval)
    else:
        rval = lr

    rval = max(rval, 1e-7)
    rval = min(rval, 1)

    return rval


def corr(x, y):

    x_ = x.mean()
    y_ = y.mean()
    return ((x - x_ * (y - y_).sum() / tensor.sqrt(((x - x_) ** 2.).sum() * ((y - y_)** 2.).sum())



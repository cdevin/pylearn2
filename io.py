import numpy
import pickle
import theano
from theano import tensor


def load_tfd(data_path, fold = 0, ds_type = 'train', shared = False):

    # unserpvised
    if fold == -1:
        path = ["{}unsupervised/TFD_unsupervised_train_unlabeled_all.pkl".format(data_path)]
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

    data_x = numpy.concatenate(data_x)
    data_y = numpy.concatenate(data_y)

    data_x = data_x.astype(theano.config.floatX) / 255.
    data_x = theano.shared(data_x)
    data_y = data_y.reshape(data_y.shape[0]) - 1
    if shared == True:
        data_y = theano.shared(data_y.astype(theano.config.floatX))
        data_y = theano.tensor.cast(data_y, 'int32')

    return data_x, data_y



def features(model, data, batch_size = 100):

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

    return numpy.concatenate(feats)




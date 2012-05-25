import numpy
import pickle
import theano


def load_tfd(fold = 0, ds_type = 'train', shared = False):

    # unserpvised
    if fold == -1:
        path = ["/data/lisatmp/rifaisal/TFD/unsupervised/TFD_unsupervised_"\
                "train_unlabeled{}.pkl".format(ind) for ind in xrange(12)]
    else:
        if ds_type not in ['train', 'valid', 'test']:
            raise NameError("wrong dataset type: {}".format(ds_type))
        if ds_type == "train":
            ds_type = 'train_labeled'

        path = ["/data/lisatmp/rifaisal/TFD/FOLD0/TFD_RAW_FOLD_0_{}{}.pkl".format(ds_type, fold)]

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



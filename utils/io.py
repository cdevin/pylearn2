import gc
import numpy
import theano
from theano import tensor
from sklearn.preprocessing import Scaler
from pylearn2.utils import serial
from noisy_encoder.scripts.train.classify import  norm
from noisy_encoder.datasets.iterator import DatasetIterator

def shared_dataset(data_x, data_y, borrow = True, cast_int = True):

    shared_x = theano.shared(numpy.asarray(data_x,
                           dtype=theano.config.floatX),
                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                           dtype=theano.config.floatX),
                 borrow=borrow)

    if cast_int:
        shared_y = tensor.cast(shared_y, 'int32')
    return shared_x, shared_y

def load_data(dataset, data_path, shuffle = False, scale = False, norm = False, fold = 0):

    print "Loading data..."
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        train_set = serial.load(data_path + 'train.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        if dataset == 'mnist':
            train_size = 50000
            nouts = 10
        elif dataset == 'cifar10':
            train_size = 45000
            nouts = 10
        elif dataset == 'cifar100':
            train_size = 40000
            nouts = 100
        else:
            raise NameError('Unknown dataset: {}'.format(dataset))

        if shuffle:
            rng = numpy.random.RandomState(23027)
            rand_idx = rng.permutation(train_set.X.shape[0])
            train_x = train_set.X[rand_idx][:train_size]
            train_y = train_set.y[rand_idx][:train_size]
            valid_x = train_set.X[rand_idx][train_size:]
            valid_y = train_set.y[rand_idx][train_size:]
        else:
            train_x = train_set.X[:train_size]
            train_y = train_set.y[:train_size]
            valid_x = train_set.X[train_size:]
            valid_y = train_set.y[train_size:]

        test_x = test_set.X
        test_y = test_set.y
        del train_set, test_set

    elif dataset in ['tfd', 'timit']:
        train_set = serial.load(data_path + 'train.pkl')
        valid_set = serial.load(data_path + 'valid.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        train_x = train_set.X[:100]
        train_y = train_set.y[:100]
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y
        del train_set, valid_set, test_set
    elif dataset in ['google']:
        train_set = serial.load(data_path + 'train.pkl')
        valid_set = serial.load(data_path + 'valid.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        train_x = train_set.X
        train_y = train_set.y
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y

        train = shared_dataset(train_x, train_y, cast_int = False)
        valid = shared_dataset(valid_x, valid_y, cast_int = False)
        test = shared_dataset(test_x, test_y, cast_int = False)

        return train, valid, test
    elif dataset in ['google_large']:
        train = DatasetIterator(data_path, 'train')
        valid_set = serial.load(data_path + 'valid.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y

        valid = shared_dataset(valid_x, valid_y, cast_int = False)
        test = shared_dataset(test_x, test_y, cast_int = False)

        return train, valid, test
    elif dataset == 'tfd_siamese':
        train_set = serial.load(data_path + 'train.pkl')
        train_set_p = serial.load(data_path + 'train_neutral.pkl')
        valid_set = serial.load(data_path + 'valid.pkl')
        valid_set_p = serial.load(data_path + 'valid_neutral.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        test_set_p = serial.load(data_path + 'test_neutral.pkl')
        train_x = train_set.X[:100]
        train_x_p = train_set_p.X[:100]
        train_y = train_set.y[:100]
        train_y_p = train_set_p.y[:100]
        valid_x = valid_set.X
        valid_x_p = valid_set_p.X
        valid_y = valid_set.y
        valid_y_p = valid_set_p.y
        test_x = test_set.X
        test_x_p = test_set_p.X
        test_y = test_set.y
        test_y_p = test_set_p.y

        train = shared_dataset(train_x, train_y)
        train_p = shared_dataset(train_x_p, train_y_p)
        valid = shared_dataset(valid_x, valid_y)
        valid_p = shared_dataset(valid_x_p, valid_y_p)
        test = shared_dataset(test_x, test_y)
        test_p = shared_dataset(test_x_p, test_y_p)

        return train, train_p, valid, valid_p, test, test_p
    elif dataset == 'tfd_siamese_variant':
        train_set = serial.load(data_path + 'train.pkl')
        train_set_p = serial.load(data_path + 'train_p.pkl')
        valid_set = serial.load(data_path + 'valid.pkl')
        valid_set_p = serial.load(data_path + 'valid_p.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        test_set_p = serial.load(data_path + 'test_p.pkl')
        train_x = train_set.X
        train_x_p = train_set_p.X
        train_y = train_set.y
        train_y_p = train_set_p.y
        valid_x = valid_set.X
        valid_x_p = valid_set_p.X
        valid_y = valid_set.y
        valid_y_p = valid_set_p.y
        test_x = test_set.X
        test_x_p = test_set_p.X
        test_y = test_set.y
        test_y_p = test_set_p.y

        train = shared_dataset(train_x, train_y)
        train_p = shared_dataset(train_x_p, train_y_p)
        valid = shared_dataset(valid_x, valid_y)
        valid_p = shared_dataset(valid_x_p, valid_y_p)
        test = shared_dataset(test_x, test_y)
        test_p = shared_dataset(test_x_p, test_y_p)

        return train, train_p, valid, valid_p, test, test_p
    elif dataset == 'google_siamese':
        train_set = serial.load(data_path[0] + 'train.pkl')
        train_set_p = serial.load(data_path[0] + 'train_neutral.pkl')
        train_x = train_set.X
        train_x_p = train_set_p.X
        train_y = train_set.y
        train_y_p = train_set_p.y

        train_siamese = shared_dataset(train_x, train_y)
        train_p = shared_dataset(train_x_p, train_y_p)

        train_set = serial.load(data_path[1] + 'train.pkl')
        valid_set = serial.load(data_path[1] + 'valid.pkl')
        test_set = serial.load(data_path[1] + 'test.pkl')
        train_x = train_set.X
        train_y = train_set.y
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y

        train = shared_dataset(train_x, train_y, cast_int = False)
        valid = shared_dataset(valid_x, valid_y, cast_int = False)
        test = shared_dataset(test_x, test_y, cast_int = False)
        test = valid
        return (train_siamese, train_p), (train, valid, test)
    elif dataset == 'tfd_siamese_mix':
        train_set = serial.load(data_path[0] + 'train.pkl')
        train_set_p = serial.load(data_path[0] + 'train_neutral.pkl')
        train_x = train_set.X[:100]
        train_x_p = train_set_p.X[:100]
        train_y = train_set.y[:100]
        train_y_p = train_set_p.y[:100]

        train_siamese = shared_dataset(train_x, train_y)
        train_p = shared_dataset(train_x_p, train_y_p)

        train_set = serial.load(data_path[1] + 'train.pkl')
        valid_set = serial.load(data_path[1] + 'valid.pkl')
        test_set = serial.load(data_path[1] + 'test.pkl')
        train_x = train_set.X[:100]
        train_y = train_set.y[:100]
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y

        train = shared_dataset(train_x, train_y, cast_int = True)
        valid = shared_dataset(valid_x, valid_y, cast_int = True)
        test = shared_dataset(test_x, test_y, cast_int = True)
        return (train_siamese, train_p), (train, valid, test)


    else:
        raise NameError('Unknown dataset: {}'.format(dataset))


    if scale:
        print "Scaling data..."
        scaler = Scaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        valid_x = scaler.transform(valid_x)
        test_x = scaler.transform(test_x)

    if norm:
        print "Normalizing..."
        train_x = numpy.vstack([norm(x) for x in train_x])
        valdd_x = numpy.vstack([norm(x) for x in valid_x])
        test_x = numpy.vstack([norm(x) for x in test_x])

    train = shared_dataset(train_x, train_y)
    valid = shared_dataset(valid_x, valid_y)
    test = shared_dataset(test_x, test_y)
    del train_x, train_y, valid_x, valid_y, test_x, test_y
    gc.collect()
    return train, valid, test


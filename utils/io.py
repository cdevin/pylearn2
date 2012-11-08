import numpy
from sklearn.preprocessing import Scaler
from pylearn2.utils import serial
from noisy_encoder.scripts.train.classify import shared_dataset, norm


def load_data(dataset, data_path, shuffle = False, scale = False, norm = False, fold = 0):

    print "Loading data..."
    if dataset in ['mnist', 'cifar10', 'cifar100']:
        train_set = serial.load(data_path + 'train.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        if dataset == 'mnist':
            train_size = 50000
            nouts = 10
        elif dataset == 'cifar10':
            train_size = 40000
            nouts = 10
        elif dataset == 'cifar100':
            train_size = 40000
            nouts = 100
        else:
            raise NameError('Unknown dataset: {}').format(dataset)

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

    elif dataset == 'timit':
        train_set = serial.load(data_path + 'train.pkl')
        valid_set = serial.load(data_path + 'valid.pkl')
        test_set = serial.load(data_path + 'test.pkl')
        nouts = 61
        train_x = train_set.X
        train_y = train_set.y
        valid_x = valid_set.X
        valid_y = valid_set.y
        test_x = test_set.X
        test_y = test_set.y
        del train_set, valid_set, test_set
    else:
        raise NameError('Unknown dataset: {}').format(dataset)


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

    return train, valid, test


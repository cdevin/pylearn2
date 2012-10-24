import numpy
import theano
import argparse
from theano import tensor
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST
from noisy_encoder.models.mlp import run_mlp as train
from sklearn.preprocessing import Scaler
from jobman.tools import DD


def shared_dataset(data_x, data_y, borrow = True):

    shared_x = theano.shared(numpy.asarray(data_x,
                           dtype=theano.config.floatX),
                 borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                           dtype=theano.config.floatX),
                 borrow=borrow)

    return shared_x, tensor.cast(shared_y, 'int32')


def kfold(train_set, test_set, n_folds, lr_vals, seed = 23007):

    rng = numpy.random.RandomState(seed)

    test_acc_list = []

    for fold in xrange(n_folds):
        print "\n\n#######"
        print "Running fold {}/{}".format(fold + 1, n_folds)
        if n_folds == 1:
            rand_idx = range(60000)
        else:
            rand_idx = rng.permutation(60000)
        train_x = train_set.X[rand_idx][:50000]
        train_y = train_set.y[rand_idx][:50000]
        valid_x = train_set.X[rand_idx][50000:]
        valid_y = train_set.y[rand_idx][50000:]
        datasets = [shared_dataset(train_x, train_y), shared_dataset(valid_x, valid_y), shared_dataset(test_set.X, test_set.y)]

        best_model = None
        best_valid_acc = -1.
        best_test_acc = -1.

        for lr in lr_vals:
            valid_acc, test_acc = train(datasets, train_x.shape[1], learning_rate = lr)
            print "For lr:{},  valid_acc:{}, test_acc:{}".format(lr, valid_acc, test_acc)

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_test_acc = test_acc

        test_acc_list.append(best_test_acc)


    test_score = numpy.mean(test_acc_list)

    print "###"
    print "Final test score: {}".format(test_score * 100.)
    return test_score

def experiment(state, channel):

    print "Loading data..."
    train_set = MNIST('train')
    test_set = MNIST('test')

    if state.model_f != None:
        print "Transforming data..."
        x = tensor.matrix()
        rep = serial.load(state.model_f)
        rep.fn = theano.function([x], rep(x))

        train_set.X = rep.perform(train_set.X)
        test_set.X  = rep.perform(test_set.X)

    if state.scale:
        print "Scaling data..."
        scaler = Scaler()
        scaler.fit(train_set.X)
        train_set.X = scaler.transform(train_set.X)
        test_set.X = scaler.transform(test_set.X)

    state.test_score = float(kfold(train_set, test_set, state.n_folds, state.lr_vals))

    return channel.COMPLETE

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'mlp classifer')
    parser.add_argument('-k', '--k_folds', default = 1, type = int, help = "K for k-fold cross-validation")
    parser.add_argument('-f', '--file', help = "transformer model file", default = None)
    parser.add_argument('-s', '--scale', action = "store_true", default = False, help = "scale data")
    parser.add_argument('-l', '--lr', help = "learning rates list", default = [0.01])
    args = parser.parse_args()

    state = DD()
    state.model_f = args.file
    state.n_folds = args.k_folds
    state.scale = args.scale
    if type(args.lr) == type("str"):
        args.lr = [float(i) for i in args.lr.split()]
    state.lr_vals = args.lr

    experiment(state, None)


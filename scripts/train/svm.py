import numpy
import theano
from theano import tensor
from pylearn2.models.svm import DenseMulticlassSVM
from pylearn2.utils import serial
from pylearn2.datasets.mnist import MNIST


def train(train_set, valid_set, C):

    train_x, train_y = train_set
    valid_x, valid_y = valid_set

    svm = DenseMulticlassSVM(C)
    svm.fit(train_x, train_y)

    valid_acc = (svm.predict(valid_y) == valid_y).mean()

    return svm, valid_acc

def kfold(train_set, test_set, n_folds, C_vals, seed = 23007):

    rng = numpy.random.RandomState(seed)

    test_acc = []

    for fold in xrange(n_folds):
        print "Runnin fold {}/{}".format(fold, n_folds)
        if n_folds > 1:
            rand_idx = rng.permutation(60000)
        else:
            rand_idx = range(60000)
        train_x = train_set.X[rand_idx][:48000]
        train_y = train_set.y[rand_idx][:48000]
        valid_x = train_set.X[rand_idx][48000:]
        valid_y = train_set.y[rand_idx][48000:]

        best_acc = -1.
        best_C = None

        for C in C_vals:
            svm_model,  valid_acc = train((train_x, train_y), (valid_x, valid_y), C)
            print "For C {},  valid_acc={}".format(C, valid_acc)

            if valid_acc > best_acc:
                best_acc = valid_acc
                best_C = C

        best_model = DenseMulticlassSVM(best_C)
        best_model.fit(train_set.X, train_set.y)
        test_pred = best_model.predict(test_set.X)
        test_acc.append((test_pred == test_set.y).mean())


    test_acc = numpy.mean(test_acc)

    print "Final test score: {}".format(test_acc)
    return test_acc



def main(model_f, n_folds, C_vals):

    print "Preparing data..."
    train_set = MNIST('train')
    test_set = MNIST('test')

    x = tensor.matrix()
    rep = serial.load(model_f)
    rep.fn = theano.function([x], rep(x))

    #train_set.X = rep.perform(train_set.X)
    #test_set.X  = rep.perform(test_set.X)

    kfold(train_set, test_set, n_folds, C_vals)

if __name__ == "__main__":

    model_f = "/RQexec/mirzameh/tmp/dln/tmp/dln_1.pkl"
    main(model_f, 1, [100])

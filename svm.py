import pickle, sys
import numpy
import theano
from theano import tensor
from time import time
from sklearn.svm import SVC
from io import load_tfd


def load_data(dataset, ds_type, fold):

    if dataset == 'tfd':
        return load_tfd(fold, ds_type)
    else:
        raise NameError("Invalid dataset: {}".format(dataset))

def features_fn(model, data, batch_size = 50):
    index = tensor.lscalar('index')
    x = tensor.matrix('x')
    return theano.function(inputs = [index],
                    outputs = model.test_encode(x),
                    givens = {x: data[index * batch_size : (index + 1) * batch_size]})

def features(model, data, batch_size, n_samples):
    func = features_fn(model, data, batch_size)
    feats = [func(i) for i in xrange(n_samples / batch_size)]
    if numpy.mod(n_samples, batch_size) != 0:
        func = features_fn(model, data[(n_samples / batch_size) * batch_size : ],
                    numpy.mod(n_samples, batch_size))
        feats.append(func(0))

    return numpy.concatenate(feats)

def classify(model,
            dataset,
            n_in,
            fold,
            c_vals,
            batch_size=600):


    train_set_x, train_set_y = load_data(dataset, 'train', fold)
    valid_set_x, valid_set_y = load_data(dataset, 'valid', fold)
    test_set_x, test_set_y = load_data(dataset, 'test', fold)

    # compute number of minibatches for training, validation and testing
    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    train_feats = features(model, train_set_x, batch_size, n_train)
    valid_feats = features(model, valid_set_x, batch_size, n_valid)
    test_feats = features(model, test_set_x, batch_size, n_test)

    valid_score = 0
    best_model = None
    for item in c_vals:
        print "checking C: {}".format(item)
        clf = SVC(scale_C = False, C = item)
        clf.fit(train_feats, train_set_y)
        score = clf.score(valid_feats, valid_set_y)
        print "score for currnet c is: {}".format(score)
        if score > valid_score:
            valid_score = score
            best_model = clf

    test_score = clf.score(test_feats, test_set_y)

    print "------\n"
    print best_model
    print "Final valid: {}, test: {}".format(valid_score, test_score)
    return valid_score, test_score

def load_model(dataset):

    return pickle.load(open(dataset, 'r'))

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
            c_vals = state.c_vals,
            batch_size= state.batch_size)


    state.valid_result = valid_result
    state.test_result = test_result

    return channel.COMPLETE

def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.dataset = "tfd"
    state.model_path = "data/tfd_30_model.pkl"
    state.nhid = 1024
    state.batch_size = 600
    state.c_vals = numpy.logspace(-3, 6, num = 20)
    state.fold = 0
    state.exp_name = 'test_run'

    experiment(state, None)

if __name__ == "__main__":

    test_experiment()

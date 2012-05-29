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


def classify(model,
            dataset,
            data_path,
            scale,
            n_in,
            fold,
            c_vals,
            batch_size=600):


    train_set_x, train_set_y = load_data(dataset, data_path, 'train', fold, scale)
    valid_set_x, valid_set_y = load_data(dataset, data_path, 'valid', fold, scale)
    test_set_x, test_set_y = load_data(dataset, data_path, 'test', fold, scale)

    # compute number of minibatches for training, validation and testing
    n_train = train_set_x.get_value(borrow=True).shape[0]
    n_valid = valid_set_x.get_value(borrow=True).shape[0]
    n_test = test_set_x.get_value(borrow=True).shape[0]

    train_feats = features(model, train_set_x, batch_size)
    valid_feats = features(model, valid_set_x, batch_size)
    test_feats = features(model, test_set_x, batch_size)

    valid_score = 0
    best_model = None
    best_c_score = None
    for item in numpy.logspace(c_vals[0],
                                c_vals[1],
                                num = c_vals[2]):
        print "checking C: {}".format(item)
        clf = SVC(scale_C = False, C = item)
        clf.fit(train_feats, train_set_y)
        score = clf.score(valid_feats, valid_set_y)
        print "score for currnet c is: {}".format(score)
        if score > valid_score:
            valid_score = score
            best_model = clf
            best_c_score = item

    test_score = clf.score(test_feats, test_set_y)

    print "------\n"
    print best_model
    print "Final valid: {}, test: {}".format(valid_score, test_score)
    return valid_score, test_score, best_c_score

    return pickle.load(open(dataset, 'r'))

def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    valid_result, test_result, c_score  = classify(model = state.model_path,
            dataset = state.dataset,
            data_path = state.data_path,
            scale = state.scale,
            n_in = state.nhid,
            fold = state.fold,
            c_vals = state.c_vals,
            batch_size= state.batch_size)


    state.valid_result = valid_result
    state.test_result = test_result
    state.c_score = c_score

    return channel.COMPLETE

def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.dataset = "tfd"
    state.dataset = "mnist"
    state.model_path = "data/mnist_196_model.pkl"
    state.data_path = DATA_PATH + "TFD/nac_layer1/"
    state.data_path = DATA_PATH + "mnist/"
    state.scale = False
    state.nhid = 1024
    state.batch_size = 600
    state.c_vals = [-3, 6, 5]
    state.fold = 0
    state.exp_name = 'test_run'

    experiment(state, None)

if __name__ == "__main__":

    test_experiment()

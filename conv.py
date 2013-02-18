import pickle, sys
import numpy
import theano
from theano import tensor
from time import time
from sklearn.svm import SVC, LinearSVC
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


def model(conv_params, images, batch_size, pool_shape):

    #conv
    image_shape = (batch_size, 1, 48, 48)

    x = tensor.matrix('x')
    conv_in = x.reshape(image_shape)
    w_conv = theano.shared(value = conv_params[0], name = 'W_conv')
    b_conv = theano.shared(value = conv_params[1], name = 'b_conv')


    conv_out = theano.tensor.nnet.conv.conv2d(input = conv_in,
                                            filters = w_conv,
                                            filter_shape = conv_params[0].shape,
                                            image_shape = image_shape)

    conv_out = tensor.nnet.sigmoid(conv_out + b_conv.dimshuffle('x', 0, 'x', 'x'))

    # pooling
    pool_size = conv_out.shape[3] / pool_shape
    outputs = []
    for i in range(pool_shape):
        for j in range(pool_shape):
            x_start = i * pool_size
            x_end = (i+1) * pool_size
            y_start = j * pool_size
            y_end = (j+1)* pool_size
            outputs.append(conv_out[:, :, x_start:x_end, y_start:y_end].max(axis = [2, 3]))


    pool_out = tensor.concatenate(outputs, axis = 1)


    index = tensor.lscalar('index')
    return  theano.function(inputs = [index],
                outputs = pool_out,
                givens = {
                    x : images[index * batch_size: (index + 1) * batch_size]})

def load_weights(path, ftype):

    if ftype == 'npy':
        return  numpy.load(path)
    else:
        return pickle.load(open(path, 'r'))

def get_features(data, conv_weights, batch_size = 200):

    # load data
    print "loading data..."
    w_conv = pickle.load(open(conv_weights, 'r'))
    b_conv = w_conv['hb']
    w_conv = w_conv['W'].reshape(1024, 1, 14 ,14)

    n_samples = data.get_value().shape[0]

    print "applying the model"
    model_f = model([w_conv, b_conv], data, batch_size, 3)
    features = [model_f(i) for i in xrange(n_samples / batch_size)]
    del model_f
    # new model for remaming data
    if numpy.mod(n_samples, batch_size) != 0:

        model_f = model([w_conv, b_conv],
                    data[(n_samples / batch_size) * batch_size :],
                    numpy.mod(n_samples, batch_size),3)
        features.append(model_f(0))

    features = numpy.concatenate(features)

    return features

def classify(conv_w_path,
                dataset,
                data_path,
                scale,
                fold,
                c_vals,
                batch_size):

    train_set_x, train_set_y = load_data(dataset, data_path, 'train', fold, scale)
    valid_set_x, valid_set_y = load_data(dataset, data_path, 'valid', fold, scale)
    test_set_x, test_set_y = load_data(dataset, data_path, 'test', fold, scale)


    train_feats = get_features(train_set_x, conv_w_path, batch_size)
    valid_feats = get_features(valid_set_x, conv_w_path, batch_size)
    test_feats = get_features(test_set_x, conv_w_path, batch_size)


    valid_score = 0
    best_model = None
    best_c_score = None
    for item in numpy.logspace(c_vals[0],
                                c_vals[1],
                                num = c_vals[2]):
        print "checking C: {}".format(item)
        clf = LinearSVC(scale_C = False, C = item)
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



def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    valid_result, test_result, c_score  = classify(conv_w_path = state.conv_w_path,
            dataset = state.dataset,
            data_path = state.data_path,
            scale = state.scale,
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
    state.conv_w_path = "data/tfd_patch_1_45_params.pkl"
    state.data_path = DATA_PATH + "TFD/raw/"
    state.scale = True
    state.nhid = 1024
    state.batch_size = 200
    state.c_vals = [3, 10, 10]
    state.fold = 0
    state.exp_name = 'test_run'

    experiment(state, None)

if __name__ == "__main__":

    test_experiment()


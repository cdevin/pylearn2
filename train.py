import pickle, sys
import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from time import time
from nact import NAENC
from io import load_tfd

def train(dataset,
                data_path,
                nhid,
                act_enc,
                act_dec,
                learning_rate,
                prob,
                l1_l,
                batch_size,
                n_epochs,
                norm,
                save_path,
                save_freq,
                exp_name):

    """
    This function run a whole experiment on ier and shows
    the reuslts
    """

    train_x, _ = load_data(dataset, data_path)
    data_shape = train_x.get_value(borrow=True).shape
    n_train_batches = data_shape[0] / batch_size

    model = NAENC(prob = prob,
                    nvis = data_shape[1],
                    nhid = nhid,
                    act_enc = act_enc,
                    act_dec = act_dec)

    # train cae
    print "... training the cae"
    t0 = time()
    train_f = model.train_funcs(train_x, batch_size)
    for epoch in xrange(n_epochs):
        cost = [train_f(index = batch_index, lr = learning_rate) for \
                batch_index in xrange(n_train_batches)]

        print "epoch {} cost: {}".format(epoch, numpy.mean(cost))

        if numpy.mod(epoch, save_freq) == 0 or epoch == (n_epochs -1):
            # save params and model
            model.save_params(save_path, "%s_%d" %(exp_name, epoch))
            model.save_model(save_path, "%s_%s" %(exp_name, epoch))

    print "done training in %0.3fs" % (time() - t0)

    return True

def load_data(dataset, data_path):

    if dataset == 'tfd':
        return load_tfd(data_path, fold = -1)
    else:
        raise NameError("Invalid dataset: {}".format(dataset))


def experiment(state, channel):
    """
    jobman experiment function
    """

    try:
        save_path = state.save_path
    except (AttributeError, KeyError) as e:
        save_path = './'

    result = train(
            dataset  = state.dataset,
            data_path = state.data_path,
            nhid = state.nhid,
            act_enc = state.act_enc,
            act_dec = state.act_dec,
            learning_rate = state.learning_rate,
            prob = state.prob,
            l1_l = state.l1_l,
            batch_size = state.batch_size,
            n_epochs = state.n_epochs,
            norm = state.norm,
            save_path = save_path,
            save_freq = state.save_freq,
             exp_name = state.exp_name)

    state.result = result

    return channel.COMPLETE


def test_experiment():
    """
    dummy function to test the module without jobman
    """

    from jobman import DD

    state = DD
    state.dataset = "tfd"
    state.data_path = "/RQexec/mirzameh/data/nact/layer1/"
    state.nhid = 100
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.learning_rate = 0.01
    state.prob = 0.3
    state.l1_l = 0.06
    state.batch_size = 50
    state.n_epochs = 102
    state.norm = False
    state.save_freq = 2
    state.exp_name = 'tfd'
    state.save_path = 'data/'

    experiment(state, None)


if __name__ == "__main__":

    test_experiment()


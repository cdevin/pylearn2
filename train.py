import pickle, sys
import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from time import time
from nact import NAENC
from util.funcs import load_tfd, load_mnist
from util.config import DATA_PATH

def train(dataset,
                data_path,
                scale,
                nhid,
                act_enc,
                act_dec,
                lr_init,
                lr_decay,
                input_corruption_level,
                hidden_corruption_level,
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

    print "loading data..."
    train_x, _ = load_data(dataset, data_path, scale)
    data_shape = train_x.get_value(borrow=True).shape
    n_train_batches = data_shape[0] / batch_size

    model = NAENC(input_corruption_level = input_corruption_level,
                    hidden_corruption_level = hidden_corruption_level,
                    nvis = data_shape[1],
                    nhid = nhid,
                    act_enc = act_enc,
                    act_dec = act_dec)

    # train cae
    print "training the model..."
    t0 = time()
    train_f = model.train_funcs(train_x, batch_size)
    for epoch in xrange(n_epochs):
        # constatn learning rate will used if lr_decay is -1
        if lr_decay == -1:
            learning_rate = lr_init
        else:
            learning_rate = lr_init * lr_decay / (lr_decay + epoch)

        cost = [train_f(index = batch_index, lr = learning_rate) for \
                batch_index in xrange(n_train_batches)]

        print "epoch {} cost: {}".format(epoch, numpy.mean(cost))
        if numpy.isnan(numpy.mean(cost)):
            print "Got NAN value"
            break

        if numpy.mod(epoch, save_freq) == 0 or epoch == (n_epochs -1):
            # save params and model
            model.save_params(save_path, "%s_%d" %(exp_name, epoch))
            model.save_model(save_path, "%s_%s" %(exp_name, epoch))

    print "done training in %0.3fs" % (time() - t0)

    return True

def load_data(dataset, data_path, scale):

    if dataset == 'tfd':
        return load_tfd(data_path, fold = -1, scale = scale, shared = True)
    if dataset == 'mnist':
        return load_mnist(data_path, ds_type = 'train', shared = True)
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
            scale = state.scale,
            nhid = state.nhid,
            act_enc = state.act_enc,
            act_dec = state.act_dec,
            lr_init = state.lr_init,
            lr_decay = state.lr_decay,
            input_corruption_level = state.input_corruption_level,
            hidden_corruption_level = state.hidden_corruption_level,
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
    #state.dataset = "mnist"
    state.data_path = DATA_PATH + "TFD/raw/"
    #state.data_path = DATA_PATH + "mnist/"
    state.scale = True
    state.nhid = 1024
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.lr_init = 0.001
    state.lr_decay = 2
    state.input_corruption_level = 0.8
    state.hidden_corruption_level = 0.5
    state.l1_l = 0.06
    state.batch_size = 50
    state.n_epochs = 200
    state.norm = False
    state.save_freq = 2
    state.exp_name = 'tfd3000'
    state.save_path = 'data/'

    experiment(state, None)


if __name__ == "__main__":

    test_experiment()


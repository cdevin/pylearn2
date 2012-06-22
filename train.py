import pickle, sys
import numpy
import theano
from theano import tensor
from theano.tensor.shared_randomstreams import RandomStreams
from time import time
from nact import NAENC
from util.funcs import load_tfd, load_mnist, learning_rate_adjuster
from util.config import DATA_PATH

def train(dataset,
                data_path,
                fold,
                scale,
                nhid,
                act_enc,
                act_dec,
                learning_rate,
                input_corruption_level,
                hidden_corruption_level,
                group_size,
                l1_reg,
                l2_reg,
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
    train_x, _ = load_data(dataset, data_path, scale, fold)
    data_shape = train_x.get_value(borrow=True).shape
    n_train_batches = data_shape[0] / batch_size

    model = NAENC(input_corruption_level = input_corruption_level,
                    hidden_corruption_level = hidden_corruption_level,
                    group_size = group_size,
                    nvis = data_shape[1],
                    nhid = nhid,
                    act_enc = act_enc,
                    act_dec = act_dec)

    # train cae
    print "training the model..."
    t0 = time()
    previous_cost = 0
    train_f = model.train_funcs(train_x, batch_size, l1_reg, l2_reg)
    for epoch in xrange(n_epochs):
        # constatn learning rate will used if lr_decay is -1
        cost = numpy.mean([train_f(index = batch_index, lr = learning_rate) for \
                batch_index in xrange(n_train_batches)])

        print "epoch {} cost: {}".format(epoch, cost)
        if numpy.isnan(numpy.mean(cost)):
            print "Got NAN value"
            break

        # adjust lr
        learning_rate = learning_rate_adjuster(cost, previous_cost, learning_rate)
        previous_cost = cost

        # save params
        if numpy.mod(epoch, save_freq) == 0 or epoch == (n_epochs -1):
            # save params and model
            model.save_params(save_path, "%s_%d" %(exp_name, epoch))
            model.save_model(save_path, "%s_%s" %(exp_name, epoch))

    print "done training in %0.3fs" % (time() - t0)

    return True

def load_data(dataset, data_path, scale, fold):

    if dataset == 'tfd':
        return load_tfd(data_path, fold = fold, scale = scale, shared = True)
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
            fold = state.fold,
            scale = state.scale,
            nhid = state.nhid,
            act_enc = state.act_enc,
            act_dec = state.act_dec,
            learning_rate = state.learning_rate,
            input_corruption_level = state.input_corruption_level,
            hidden_corruption_level = state.hidden_corruption_level,
            group_size = state.group_size,
            l1_reg = state.l1_reg,
            l2_reg = state.l2_reg,
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
    state.data_path = DATA_PATH + "TFD/raw/"
    #state.data_path = '/RQexec/mirzameh/data/mnist/'
    #state.fold = "tfd_unlabled_14x14_patches"
    state.fold = -1
    state.scale = True
    state.nhid = 1024
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.learning_rate = 0.01
    state.input_corruption_level = 0.0
    state.hidden_corruption_level = 0.7
    state.group_size = -1
    state.l1_reg = 0.5
    state.l2_reg = 0.5
    state.batch_size = 50
    state.n_epochs = 100
    state.norm = False
    state.save_freq = 3
    state.exp_name = 'tfd_test'
    state.save_path = 'data/'

    experiment(state, None)


if __name__ == "__main__":

    test_experiment()


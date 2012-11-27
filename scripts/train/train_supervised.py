import os
import numpy
import argparse
from jobman.tools import DD
from utils.config import get_data_path, get_result_path
from noisy_encoder.utils.io import load_data
from noisy_encoder.training_algorithms.sgd import sgd
from noisy_encoder.models.mlp import MLP
from noisy_encoder.models.conv import LeNetLearner
from noisy_encoder.models.siamese import Siamese
from noisy_encoder.utils.corruptions import BinomialCorruptorScaled
from theano.tensor.shared_randomstreams import RandomStreams


RESULT_PATH = get_result_path()
DATA_PATH = get_data_path()

def load_model(state, numpy_rng, theano_rng):
    if state.model == 'mlp':
        return MLP(numpy_rng = numpy_rng,
                theano_rng = theano_rng,
                n_units = state.n_units,
                gaussian_corruption_levels = state.gaussian_corruption_levels,
                binomial_corruption_levels = state.binomial_corruption_levels,
                group_sizes = state.group_sizes,
                n_outs = state.nouts,
                act_enc = state.act_enc,
                irange = state.irange,
                bias_init = state.bias_init,
                group_corruption_levels = state.group_corruption_levels)
    elif state.model == 'conv':
        return Conv(rng = numpy_rng,
                theano_rng = theano_rng,
                image_shapes = state.image_shapes,
                nkerns = state.nkerns,
                filter_shapes = state.filter_shapes,
                poolsizes = state.poolsizes,
                binomial_corruption_levels = state.binomial_corruption_levels,
                gaussian_corruption_levels = state.gaussian_corruption_levels,
                nhid = state.nhid,
                nout = state.nouts,
                activation = state.activation,
                batch_size = state.batch_size)
    elif state.model == 'new_conv':
        return LeNetLearner(
                image_shape = state.image_shape,
                kernel_shapes = state.kernel_shapes,
                nchannels = state.nchannels,
                pool_shapes = state.pool_shapes,
                batch_size = state.batch_size,
                conv_act = state.conv_act,
                mlp_act = state.mlp_act,
                mlp_input_corruption_levels = state.mlp_input_corruption_levels,
                mlp_hidden_corruption_levels = state.mlp_hidden_corruption_levels,
                mlp_nunits = state.mlp_nunits,
                n_outs = state.n_outs,
                irange = state.irange,
                bias_init = state.bias_init,
                rng = numpy_rng)
    elif state.model == 'siamese':
        return Siamese(numpy_rng = numpy_rng,
                theano_rng = theano_rng,
                image_topo = state.image_topo,
                base_model = state.base_model,
                n_units = state.n_units,
                input_corruption_levels = state.input_corruption_levels,
                hidden_corruption_levels = state.hidden_corruption_levels,
                n_outs = state.nouts,
                act_enc = state.act_enc,
                irange = state.irange,
                bias_init = state.bias_init,
                method = state.method,
                fine_tune = state.fine_tune)
    else:
        raise NameError("Unknown model: {}".format(state.model))

def experiment(state, channel):

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    datasets = load_data(state.dataset,
                        state.data_path,
                        state.shuffle,
                        state.scale,
                        state.norm,
                        state.fold)
    model = load_model(state, numpy_rng, theano_rng)
    state.test_score, state.valid_score = sgd(model = model,
                                datasets = datasets,
                                learning_rate_init = state.lr,
                                training_epochs = state.nepochs,
                                batch_size = state.batch_size,
                                w_l1_ratio = state.w_l1_ratio,
                                act_l1_ratio = state.act_l1_ratio,
                                lr_shrink_time = state.lr_shrink_time,
                                lr_dc_rate = state.lr_dc_rate,
                                save_frequency = state.save_frequency,
                                save_name = state.save_name,
                                enable_momentum = state.enable_momentum,
                                init_momentum = state.init_momentum,
                                final_momentum = state.final_momentum,
                                momentum_inc_start = state.momentum_inc_start,
                                momentum_inc_end = state.momentum_inc_end)

    return channel.COMPLETE

def cifar10_experiment():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/")
    state.nouts = 10
    state.scale = False
    state.dataset = 'cifar10'
    state.norm = False
    state.nepochs = 1000
    state.model = 'mlp'
    state.act_enc = "rectifier"
    state.lr = 0.01
    state.lr_shrink_time = 1000
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 100
    state.batch_size = 100
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.05
    state.shuffle = False
    state.n_units = [32*32*3, 1000, 1000, 1000]
    state.gaussian_corruption_levels = [0.0, 0.0, 0.0, 0.0]
    state.binomial_corruption_levels = [0.0, 0.0, 0.5, 0.5]
    #state.group_corruption_levels = [0.0, 0.0, 0.5] # set this to None to stop group training
    state.group_corruption_levels = None
    state.group_sizes = [128, 128, 128]
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/cifar10/mlp_2.pkl")
    state.fold = 0

    experiment(state, None)

def cifar100_experiment():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "cifar100/pca/")
    state.nouts = 100
    state.scale = False
    state.dataset = 'cifar100'
    state.norm = False
    state.nepochs = 1000
    state.model = 'mlp'
    state.act_enc = "rectifier"
    state.lr = 0.0001
    state.lr_shrink_time = 300
    state.lr_dc_rate = 0.001
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 70
    state.batch_size = 200
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.1
    state.bias_init = 0.0
    state.shuffle = False
    #state.n_units = [32*32*3, 1024, 1024, 1024, 1024]
    state.n_units = [512, 10000, 1000]
    state.gaussian_corruption_levels = [0.0, 0.0, 0.0]
    state.binomial_corruption_levels = [0.0, 0.9, 0.0]
    #state.group_corruption_levels = [0.0, 0.0, 0.5] # set this to None to stop group training
    state.group_corruption_levels = None
    state.group_sizes = [None, 1000, None]
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/cifar100/mlp.pkl")
    state.fold = 0

    experiment(state, None)

def mnist_experiment():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "mnist/pylearn2/")
    state.nouts = 100
    state.scale = True
    state.dataset = 'mnist'
    state.norm = False
    state.nepochs = 1000
    state.model = 'mlp'
    state.act_enc = "rectifier"
    state.lr = 0.01
    state.lr_shrink_time = 50
    state.lr_dc_rate = 0.001
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 70
    state.batch_size = 200
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.1
    state.shuffle = False
    state.n_units = [28 * 28, 1024, 1024]
    state.gaussian_corruption_levels = [0.3, 0.0, 0.0]
    state.binomial_corruption_levels = [0.5, 0.5, 0.0]
    state.group_corruption_levels = None
    state.group_sizes = [128, 128, 128]
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/mnist/mlp.pkl")
    state.fold = 0

    experiment(state, None)

def tfd_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/".format(state.fold))
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.nepochs = 1000
    state.lr = 0.01
    state.lr_shrink_time = 100
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 30
    state.momentum_inc_end = 70
    state.batch_size = 200
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.save_frequency = 50
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/conv.pkl")

    # model params
    state.model = 'conv'
    state.activation = "tanh"
    state.nouts = 7
    state.image_shapes = [(48, 48), (21, 21), (9, 9), (3, 3)]
    state.nkerns =  [1, 20, 50, 100]
    state.filter_shapes =  [(20, 1, 7, 7), (50, 20, 4, 4), (100, 50, 4, 4)]
    state.poolsizes =  [(2, 2), (2, 2), (2, 2)]
    state.gaussian_corruption_levels = [0.0, 0.0, 0.0, 0.0, 0.0]
    state.binomial_corruption_levels = [0.0, 0.5, 0.5, 0.5, 0.5]
    state.nhid = 500
    state.irange = 0.01

    experiment(state, None)

def tfd_newconv_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/TFD/pylearn2_aug/{}/".format(state.fold))
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.nepochs = 1000
    state.lr = 0.01
    state.lr_shrink_time = 70
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 30
    state.momentum_inc_end = 70
    state.batch_size = 100
    state.w_l1_ratio = 0.0005
    state.act_l1_ratio = 0.0
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/conv.pkl")

    # model params
    state.model = 'new_conv'
    state.image_shape = [48, 48]
    state.kernel_shapes = [(7,7), (4, 4), (4, 4)]
    state.nchannels = [1, 20, 50, 80]
    state.pool_shapes = [(2,2), (2, 2), (2, 2)]
    state.conv_act = "rectifier"
    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [None, None]
    state.mlp_hidden_corruption_levels = [0.5, 0.5]
    state.mlp_nunits = [1000, 500]
    state.n_outs = 7
    state.bias_init = 0.1
    state.irange = 0.1

    experiment(state, None)

def siamese_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd_siamese'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/TFD/siamese/{}/".format(state.fold))
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.nepochs = 1000
    state.lr = 0.01
    state.lr_shrink_time = 100
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 30
    state.momentum_inc_end = 70
    state.batch_size = 100
    state.w_l1_ratio = 0.0000
    state.act_l1_ratio = 0.0
    state.save_frequency = 50
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/siamese.pkl")

    # model params
    state.model = 'siamese'
    state.method = 'diff'
    state.fine_tune = False
    #state.base_model = os.path.join(RESULT_PATH, "models/tfd_conv/{}.pkl".format(state.fold))
    state.base_model = os.path.join(RESULT_PATH, "naenc/tfd/conv.pkl")
    state.image_topo = (state.batch_size, 48, 48, 1)
    state.n_units = [500, 1000, 500]
    state.input_corruption_levels = [None, None, None]
    state.hidden_corruption_levels = [0.5, 0.5, 0.5]
    state.nouts = 6
    state.act_enc = "rectifier"
    state.irange = 0.1
    state.bias_init = 0.1

    experiment(state, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'supervised trainer')
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10',
        'cifar100', 'timit', 'tfd', 'tfd_new_conv', 'siamese'], required = True)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        mnist_experiment()
    elif args.dataset == 'cifar10':
        cifar10_experiment()
    elif args.dataset == 'cifar100':
        cifar100_experiment()
    elif args.dataset == 'tfd':
        tfd_experiment()
    elif args.dataset == 'mnsit':
        mnist_experiment()
    elif args.dataset == 'siamese':
        siamese_experiment()
    elif args.dataset == 'tfd_new_conv':
        tfd_newconv_experiment()

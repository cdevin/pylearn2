import os
import numpy
import argparse
from jobman.tools import DD
from my_utils.config import get_data_path, get_result_path
from noisy_encoder.utils.io import load_data
from noisy_encoder.training_algorithms.sgd import sgd, sgd_mix, sgd_large
from noisy_encoder.models.conv import LeNetLearner, LeNetLearnerMultiCategory
from noisy_encoder.models.siamese import Siamese, SiameseVariant, SiameseMix, SiameseMixSingleCategory
from noisy_encoder.utils.corruptions import BinomialCorruptorScaled
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


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
        return LeNetLearner(
                conv_layers = state.conv_layers,
                batch_size = state.batch_size,
                mlp_act = state.mlp_act,
                mlp_input_corruption_levels = state.mlp_input_corruption_levels,
                mlp_hidden_corruption_levels = state.mlp_hidden_corruption_levels,
                mlp_nunits = state.mlp_nunits,
                n_outs = state.n_outs,
                irange = state.irange,
                bias_init = state.bias_init,
                random_filters = state.random_filters,
                th_rng = theano_rng,
                np_rng = numpy_rng)
    elif state.model == 'google_conv':
        return LeNetLearnerMultiCategory(
                image_shape = state.image_shape,
                kernel_shapes = state.kernel_shapes,
                nchannels = state.nchannels,
                pool_shapes = state.pool_shapes,
                batch_size = state.batch_size,
                conv_act = state.conv_act,
                normalize_params = state.normalize_params,
                mlp_act = state.mlp_act,
                mlp_input_corruption_levels = state.mlp_input_corruption_levels,
                mlp_hidden_corruption_levels = state.mlp_hidden_corruption_levels,
                mlp_nunits = state.mlp_nunits,
                n_outs = state.n_outs,
                irange = state.irange,
                bias_init = state.bias_init,
                rng = numpy_rng)
    elif state.model == 'google_siamese':
        return SiameseMix(numpy_rng = numpy_rng,
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
    elif state.model == 'tfd_siamese_mix':
        return SiameseMixSingleCategory(numpy_rng = numpy_rng,
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
    elif state.model == 'siamese_variant':
        return SiameseVariant(numpy_rng = numpy_rng,
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
    if state.train_alg == "sgd":
        train_alg = sgd
    elif state.train_alg == "sgd_mix":
        train_alg = sgd_mix
    elif state.train_alg == "sgd_large":
        train_alg = sgd_large
    else:
        raise NameError("Unknown training algorithms: {}".format(train_alg))
    state.test_score, state.valid_score = train_alg(model = model,
                                datasets = datasets,
                                training_epochs = state.nepochs,
                                batch_size = state.batch_size,
                                coeffs = state.coeffs,
                                lr_params = state.lr_params,
                                save_frequency = state.save_frequency,
                                save_name = state.save_name,
                                enable_momentum = state.enable_momentum,
                                momentum_params = state.momentum_params)

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

def cifar10_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'cifar10'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "cifar10_conv/pylearn2/")
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = "sgd"
    state.nepochs = 300
    state.lr_params = {'shrink_time': 10, 'init_value' : 0.005, 'dc_rate' : 0.001}
    state.enable_momentum = True
    state.momentum_params = {'inc_start' : 30, 'inc_end' : 70, 'init_value' : 0.5, 'final_value' : 0.9}
    state.batch_size = 50
    state.w_l1_ratio = 0.000
    state.act_l1_ratio = 0.0
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/tfd_gpu.pkl")
    state.coeffs = {'w_l1' : 0.0, 'w_l2' : 0e-06}
    # model params
    state.model = 'conv'
    state.conv_layers = [
                {'name' : 'Convolution',
                    'params' : {'image_shape' : [32, 32],
                            'kernel_shape' : [5, 5],
                            'num_channels_input' : 3,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : [28, 28],
                        'num_channels' : 64,
                        'pool_shape' : (3, 3),
                        'pool_stride' : (2, 2)}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [14, 14],
                            'batch_size' : state.batch_size,
                            'num_channels' : 64,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : [14, 14],
                            'kernel_shape' : [5, 5],
                            'num_channels_input' : 64,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : [10, 10],
                        'num_channels' : 64,
                        'pool_shape' : (3, 3),
                        'pool_stride' : (2, 2)}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [5, 5],
                            'batch_size' : state.batch_size,
                            'num_channels' : 64,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}}]

    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [None, None]
    state.mlp_hidden_corruption_levels = [0.5, 0.5]
    state.mlp_nunits = [1000, 500]
    state.n_outs = 10
    state.bias_init = 0.1
    state.irange = 0.1
    state.random_filters = True

    experiment(state, None)

def tfd_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/".format(state.fold))
    #state.data_path = os.path.join(DATA_PATH, "faces/tfd_lisa/pylearn2/")
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = "sgd"
    state.nepochs = 500
    state.lr_params = {'shrink_time': 50, 'init_value' : 0.005, 'dc_rate' : 0.001}
    state.enable_momentum = True
    state.momentum_params = {'inc_start' : 70, 'inc_end' : 120, 'init_value' : 0.5, 'final_value' : 0.9}
    state.batch_size = 20
    state.w_l1_ratio = 0.000
    state.act_l1_ratio = 0.0
    state.save_frequency = 100
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/tfd_gpu.pkl")
    state.coeffs = {'w_l1' : 0.0, 'w_l2' : 0.0}
    # model params
    state.model = 'conv'
    state.conv_layers = [
             {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [48, 48],
                            'batch_size' : state.batch_size,
                            'num_channels' : 1,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : [48, 48],
                            'kernel_shape' : [7, 7],
                            'num_channels' : 1,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                #{'name' : 'StochasticMaxPool',
                    #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : (3, 3),
                        #'pool_stride' : (2, 2)}},
                {'name' : 'MaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : (3, 3),}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : None,
                            'batch_size' : state.batch_size,
                            'num_channels' : None,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75
                            }},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : None,
                            'kernel_shape' : [5, 5],
                            'num_channels' : None,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                #{'name' : 'StochasticMaxPool',
                    #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : (3, 3),
                        #'pool_stride' : (2, 2)}},
                {'name' : 'MaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : (2, 2),}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : None,
                            'batch_size' : state.batch_size,
                            'num_channels' :None ,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75
                            }}]
                #{'name' : 'Convolution',
                    #'params' : {'image_shape' : None,
                            #'kernel_shape' : [4, 4],
                            #'num_channels' : None,
                            #'num_channels_output' : 64,
                            #'batch_size' : state.batch_size,
                            #'act_enc' : 'sigmoid',}},
                #{'name' : 'StochasticMaxPool',
                    #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : (3, 3),
                        #'pool_stride' : (2, 2)}},
                #{'name' : 'MaxPool',
                    #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : (2, 2),}}]
                #{'name' : 'LocalResponseNormalize',
                    #'params' : {'image_shape' : None,
                            #'batch_size' : state.batch_size,
                            #'num_channels' : None,
                            #'n' : 4,
                            #'k' : 1,
                            #'alpha' : 0e-04,
                            #'beta' : 0.75
                            #}}]

    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [None, None]
    state.mlp_hidden_corruption_levels = [0.5, 0.5]
    state.mlp_nunits = [1200]
    state.n_outs = 7
    state.bias_init = 0.1
    state.irange = 0.1
    state.random_filters = True

    experiment(state, None)

def google_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'google'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/google_tfd_lisa_aug/pylearn2/")
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = 'sgd'
    state.nepochs = 1000
    state.lr_params = {'shrink_time': 10, 'init_value' : 0.005, 'dc_rate' : 0.001}
    state.enable_momentum = True
    state.momentum_params = {'inc_start' : 30, 'inc_end' : 70, 'init_value' : 0.5, 'final_value' : 0.9}
    state.batch_size = 1
    state.save_frequency = 10
    state.save_name = os.path.join(RESULT_PATH, "naenc/google/conv_aug_gpu.pkl")
    state.coeffs = {'w_l1' : 0.0, 'w_l2' : 1e-04}

    # model params
    state.model = 'google_conv'
    state.image_shape = [48, 48]
    state.kernel_shapes = [(7,7), (5, 5)]
    state.nchannels = [1, 64, 128]
    state.pool_shapes = [(3,3), (2, 2), (2, 2)]
    state.normalize_params = [{'n':5, 'k':1, 'alpha':0e-04, 'beta':0.75, 'image_size':42, 'nkernels':64 },
            {'n':4, 'k':1, 'alpha':0e-04, 'beta':0.75, 'image_size':10, 'nkernels':128}]
    state.conv_act = "rectifier"
    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [None, None]
    state.mlp_hidden_corruption_levels = [0.5, 0.0]
    state.mlp_nunits = [1000, 7]
    state.n_outs = 7
    state.bias_init = 0.1
    state.irange = 0.1

    experiment(state, None)

def google_large_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'google_large'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/google_tfd_lisa_aug/pylearn2/")
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = 'sgd_large'
    state.nepochs = 1000
    state.lr_params = {'shrink_time': 10, 'init_value' : 0.0005, 'dc_rate' : 0.001}
    state.enable_momentum = True
    state.momentum_params = {'inc_start' : 30, 'inc_end' : 70, 'init_value' : 0.5, 'final_value' : 0.9}
    state.batch_size = 10
    state.save_frequency = 1
    state.save_name = os.path.join(RESULT_PATH, "naenc/google/conv_aug_gpu.pkl")
    state.coeffs = {'w_l1' : 0.0, 'w_l2' : 1e-06}

    # model params
    state.model = 'google_conv'
    state.image_shape = [48, 48]
    state.kernel_shapes = [(7,7), (5, 5)]
    state.nchannels = [1, 64, 64]
    state.pool_shapes = [(3,3), (2, 2), (2, 2)]
    state.normalize_params = [{'n':4, 'k':1, 'alpha':0e-04, 'beta':0.75, 'image_size':42, 'nkernels':64 },
            {'n':4, 'k':1, 'alpha':0e-04, 'beta':0.75, 'image_size':10, 'nkernels':64}]
    state.conv_act = "rectifier"
    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [None, None]
    state.mlp_hidden_corruption_levels = [0.5, 0.5]
    state.mlp_nunits = [1000, 7]
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
    state.save_frequency = 10
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/siamese.pkl")
    state.coeffs = {'w_l1' : 0.0}

    # model params
    state.model = 'siamese'
    state.method = 'diff'
    state.fine_tune = False
    #state.base_model = os.path.join(RESULT_PATH, "models/tfd_conv/{}.pkl".format(state.fold))
    state.base_model = os.path.join(RESULT_PATH, "naenc/tfd/conv_aug_gpu.pkl")
    state.image_topo = (state.batch_size, 48, 48, 1)
    state.n_units = [500, 1000, 500]
    state.input_corruption_levels = [None, None, None]
    state.hidden_corruption_levels = [0.5, 0.5, 0.5]
    state.nouts = 6
    state.act_enc = "rectifier"
    state.irange = 0.1
    state.bias_init = 0.1

    experiment(state, None)

def siamese_variant_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd_siamese_variant'
    state.fold = 0
    state.data_path = os.path.join(DATA_PATH, "faces/TFD/pylearn2_rotate/{}/".format(state.fold))
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.nepochs = 1000
    state.lr = 0.0001
    state.lr_shrink_time = 100
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 30
    state.momentum_inc_end = 70
    state.batch_size = 20
    state.w_l1_ratio = 0.0000
    state.act_l1_ratio = 0.0
    state.save_frequency = 50
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/siamese.pkl")
    state.coeffs = {'nll_p' : 0.1, 'jacob' : 0.01, 'mlp_l1': 0.00005, 'reg_l1': 0.001}



    # model params
    state.model = 'siamese_variant'
    state.method = 'diff'
    state.fine_tune = False
    #state.base_model = os.path.join(RESULT_PATH, "models/tfd_conv/{}.pkl".format(state.fold))
    state.base_model = os.path.join(RESULT_PATH, "naenc/google/conv_gpu.pkl")
    state.image_topo = (state.batch_size, 48, 48, 1)
    state.n_units = [500, 1000]
    state.input_corruption_levels = [None, None, None]
    state.hidden_corruption_levels = [0.0, 0.0, 0.0]
    state.nouts = [6, 20]
    state.act_enc = "rectifier"
    state.irange = 0.1
    state.bias_init = 0.1

    experiment(state, None)

def google_siamese_experiment():

    state = DD()

    # train params
    state.dataset = 'google_siamese'
    state.fold = 0
    state.data_path = [os.path.join(DATA_PATH, "faces/TFD/siamese/all/"),
            os.path.join(DATA_PATH, "faces/google_tfd_lisa_aug/pylearn2/")]
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = 'sgd_mix'
    state.nepochs = 1000
    state.lr = [0.005, 0.005]
    state.lr_shrink_time = 100
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 30
    state.momentum_inc_end = 70
    state.batch_size = 1
    state.w_l1_ratio = 0.0000
    state.act_l1_ratio = 0.0
    state.save_frequency = 1
    state.save_name = os.path.join(RESULT_PATH, "naenc/google/siamese.pkl")
    state.coeffs = {'conv_w_l1' : 0.000001, 'conv_w_l2' : 0.000001}

    # model params
    state.model = 'google_siamese'
    state.method = 'diff'
    state.fine_tune = False
    state.base_model = os.path.join(RESULT_PATH, "naenc/google/conv_aug_gpu.pkl")
    state.image_topo = (state.batch_size, 48, 48, 1)
    state.n_units = [1000, 500]
    state.input_corruption_levels = [None, None, None]
    state.hidden_corruption_levels = [0.5, 0.0, 0.0]
    state.nouts = 7
    state.act_enc = "sigmoid"
    state.irange = 0.1
    state.bias_init = 0.1

    experiment(state, None)

def tfd_siamese_mix_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd_siamese_mix'
    state.fold = 0
    state.data_path = [os.path.join(DATA_PATH, "faces/TFD/siamese/all/"),
            os.path.join(DATA_PATH, "faces/tfd_lisa/pylearn2/")]
    state.scale = False
    state.norm = False
    state.shuffle = False
    state.train_alg = 'sgd_mix'
    state.nepochs = 1000
    state.lr_params = [{'shrink_time': 10, 'init_value' : 0.0005, 'dc_rate' : 0.001},
            {'shrink_time': 10, 'init_value' : 0.005, 'dc_rate' : 0.001}]

    state.enable_momentum = True
    state.momentum_params = {'inc_start' : 30, 'inc_end' : 70, 'init_value' : 0.5, 'final_value' : 0.9}
    state.batch_size = 1
    state.w_l1_ratio = 0.0000
    state.act_l1_ratio = 0.0
    state.save_frequency = 1
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/siamese_mix_cpu2.pkl")
    state.coeffs = {'conv_w_l1' : 0.0, 'conv_w_l2' : 0.00001}

    # model params
    state.model = 'tfd_siamese_mix'
    state.method = 'diff'
    state.fine_tune = False
    state.base_model = os.path.join(RESULT_PATH, "naenc/tfd/lisa_conv_cpu2.pkl")
    state.image_topo = (state.batch_size, 48, 48, 1)
    state.n_units = [1000]
    state.input_corruption_levels = [None, None, None]
    state.hidden_corruption_levels = [0.5, 5.0, 0.0]
    state.nouts = 7
    state.act_enc = "rectifier"
    state.irange = 0.1
    state.bias_init = 0.1

    experiment(state, None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'supervised trainer')
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10',
        'cifar100', 'timit', 'tfd', 'tfd_conv', 'siamese', 'siamese_variant',
        'conv_google', 'siamese_google', 'tfd_siamese_mix', 'google_large_conv',
        'cifar10_conv'], required = True)
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
    elif args.dataset == 'siamese_variant':
        siamese_variant_experiment()
    elif args.dataset == 'tfd_conv':
        tfd_conv_experiment()
    elif args.dataset == 'conv_google':
        google_conv_experiment()
    elif args.dataset == 'siamese_google':
        google_siamese_experiment()
    elif args.dataset == 'tfd_siamese_mix':
        tfd_siamese_mix_experiment()
    elif args.dataset == 'google_large_conv':
        google_large_conv_experiment()
    elif args.dataset == 'cifar10_conv':
        cifar10_conv_experiment()

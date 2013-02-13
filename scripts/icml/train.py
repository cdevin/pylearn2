import os, shutil
import argparse
import numpy
from pylearn2.config import yaml_parse
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

train_yaml = """!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.svhn.SVHN {
        which_set: 'splited_train',
        path: %(data_path)s
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 128,
        layers: [
                 !obj:galatea.mlp.ConvLinearC01B {
                     layer_name: 'h0',
                     pad: 0,
                     detector_channels: %(num_channels_1)i,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: %(max_kernel_norm_1)f,
                     W_lr_scale: %(W_lr_scale_1)f,
                     b_lr_scale: %(b_lr_scale_1)f,
                     tied_b: 1
                 },
                 !obj:galatea.mlp.ConvLinearC01B {
                     layer_name: 'h1',
                     pad: 3,
                     detector_channels: %(num_channels_2)i,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: %(max_kernel_norm_2)f,
                     W_lr_scale: %(W_lr_scale_2)f,
                     b_lr_scale: %(b_lr_scale_2)f,
                     tied_b: 1
                 },
                 !obj:galatea.mlp.ConvLinearC01B {
                     pad: 3,
                     layer_name: 'h2',
                     detector_channels: %(num_channels_3)i,
                     channel_pool_size: 4,
                     kernel_shape: [5, 5],
                     pool_shape: [2, 2],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: %(max_kernel_norm_3)f,
                     W_lr_scale: %(W_lr_scale_3)f,
                     b_lr_scale: %(b_lr_scale_3)f,
                     tied_b: 1
                     #use_bias: 0
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     #max_col_norm: 3.873,
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     irange: .005
                 }
                ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [32, 32],
            num_channels: 3,
            axes: ['b', 0, 1, 'c'],
        },
        dropout_include_probs: [ .5, .5, .5, 1 ],
        dropout_input_include_prob: .8,
        dropout_input_scale: 1.,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: %(learning_rate)f,
        init_momentum: .5,
        monitoring_dataset:
            {
                #'train' : *train,
                'valid' : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'valid',
                              path: %(data_path)s
                          },
                'test' : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'test',
                              path: %(data_path)s
                          }
            },
        cost: !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
        },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria : [!obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : %(max_epochs)i
            },
            !obj:pylearn2.termination_criteria.MonitorBased {
                channel_name: "valid_y_misclass",
                prop_decrease: 0.,
                N: %(termination_paitence)i}
            ]
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "%(best_save_path)s"
        }, !obj:noisy_encoder.utils.best_params.MonitorBasedBest {
            channel_name: 'valid_y_misclass',
            save_channel_names: ['valid_y_misclass', 'test_y_misclass']
        }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: %(momentum_start)i,
            saturate: %(momentum_saturate)i,
            final_momentum: %(final_momentum)f
        }, !obj:pylearn2.training_algorithms.sgd.LinearDecayOverEpoch {
            start: %(lr_decay_start)i,
            saturate: %(lr_deccay_saturate)i,
            decay_factor: %(lr_decay_factor)f
        }
    ],
    save_path: "%(save_path)s",
    save_freq: 1
}"""


sp_train_yaml = """!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.cifar10.CIFAR10 {
        toronto_prepro: 1,
        which_set: 'train',
        one_hot: 1,
        axes: ['c', 0, 1, 'b'],
        start: 0,
        stop: 40000
    },
    model: !obj:noisy_encoder.models.convlinear.MLP {
        batch_size: 50,
        layers: [
                 !obj:noisy_encoder.models.convlinear.ConvLinearC01BStochastic {
                     layer_name: 'h0',
                     pad: 4,
                     tied_b: %(tied_b_1)i,
                     W_lr_scale: %(W_lr_scale_1)f,
                     b_lr_scale: %(b_lr_scale_1)f,
                     detector_channels: %(num_channels_1)i,
                     channel_pool_size: %(channel_pool_size_1)i,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: .9,
                 },
                 !obj:noisy_encoder.models.convlinear.ConvLinearC01BStochastic {
                     layer_name: 'h1',
                     pad: 3,
                     tied_b: %(tied_b_2)i,
                     W_lr_scale: %(W_lr_scale_2)f,
                     b_lr_scale: %(b_lr_scale_2)f,
                     detector_channels: %(num_channels_2)i,
                     channel_pool_size: %(channel_pool_size_2)i,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                 },
                 !obj:noisy_encoder.models.convlinear.ConvLinearC01BStochastic {
                     pad: 3,
                     layer_name: 'h2',
                     tied_b: %(tied_b_3)i,
                     W_lr_scale: %(W_lr_scale_3)f,
                     b_lr_scale: %(b_lr_scale_3)f,
                     detector_channels: %(num_channels_3)i,
                     channel_pool_size: %(channel_pool_size_3)i,
                     kernel_shape: [5, 5],
                     pool_shape: [2, 2],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                 },
                !obj:galatea.mlp.MaxPoolRectifiedLinear {
                    layer_name: 'h3',
                    irange: .005,
                    detector_layer_dim: 1200,
                    pool_size: 5,
                    max_col_norm: 1.9
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 10,
                     irange: .005
                 }
                ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [32, 32],
            num_channels: 3,
            axes: ['c', 0, 1, 'b'],
        },
        dropout_include_probs: [ 1, 1, 1, 1, 1],
        dropout_input_include_prob: .8,
        dropout_input_scale: 1.,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: %(learning_rate)f,
        init_momentum: %(init_momentum)f,
        monitoring_dataset:
            {
                #'train' : *train,
                'valid' : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                              toronto_prepro: 1,
                              axes: ['c', 0, 1, 'b'],
                              which_set: 'train',
                              one_hot: 1,
                              start: 40000,
                              stop:  50000
                          },
                #'test'  : !obj:pylearn2.datasets.cifar10.CIFAR10 {
                #              which_set: 'test',
                #              gcn: 55.,
                #              one_hot: 1,
                #          }
            },
        cost: !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
        },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria : [!obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : %(max_epochs)i
            },
            !obj:pylearn2.termination_criteria.MonitorBased {
                channel_name: "valid_y_misclass",
                prop_decrease: 0.,
                N: %(termination_paitence)i}
            ]
        },
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "%(best_save_path)s"
        }, !obj:noisy_encoder.utils.best_params.MonitorBasedBest {
            channel_name: 'valid_y_misclass',
            save_channel_names: ['valid_y_misclass']
        }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: %(momentum_start)i,
            saturate: %(momentum_saturate)i,
            final_momentum: %(final_momentum)f
        }, !obj:pylearn2.training_algorithms.sgd.LinearDecayOverEpoch {
            start: %(lr_decay_start)i,
            saturate: %(lr_deccay_saturate)i,
            decay_factor: %(lr_decay_factor)f
        }
    ],
    save_path: "%(save_path)s",
    save_freq: 1
}"""

sp_soft_yaml = """
!obj:pylearn2.train.Train {
    dataset: &train  !obj:pylearn2.datasets.tfd.TFD {
        which_set: 'train',
        one_hot: 1,
        fold: %(fold)i,
        axes: ['c', 0, 1, 'b'],
        preprocessor: !obj:pylearn2.datasets.preprocessing.GlobalContrastNormalization {}
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 50,
        layers: [
                 !obj:noisy_encoder.models.convlinear.ConvLinearStochasticSoftmaxPoolC01B {
                     layer_name: 'h0',
                     W_lr_scale: %(W_lr_scale_1)f,
                     b_lr_scale: %(b_lr_scale_1)f,
                     pad: 0,
                     detector_channels: %(num_channels_1)i,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: .9,
                 },
                !obj:noisy_encoder.models.convlinear.ConvLinearStochasticSoftmaxPoolC01B {
                     layer_name: 'h1',
                     W_lr_scale: %(W_lr_scale_2)f,
                     b_lr_scale: %(b_lr_scale_2)f,
                     pad: 3,
                     detector_channels: %(num_channels_2)i,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                 },
                 !obj:pylearn2.models.mlp.RectifiedLinear {
                    layer_name: 'h2',
                    irange: .005,
                    dim: 1200,
                    max_col_norm: 1.9
                 },
                 !obj:pylearn2.models.mlp.Softmax {
                     #max_col_norm: 3.873,
                     max_col_norm: 1.9365,
                     layer_name: 'y',
                     n_classes: 7,
                     irange: .005
                 }
                ],
        input_space: !obj:pylearn2.space.Conv2DSpace {
            shape: [48, 48],
            num_channels: 1,
            axes: ['c', 0, 1, 'b'],
        },
        dropout_include_probs: [ 1, 1, .5, 1 ],
        dropout_input_include_prob: %(dropout_inp)f,
        dropout_input_scale: 1.,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        learning_rate: %(learning_rate)f,
        init_momentum: %(init_momentum)f,
        monitoring_dataset:
            {
                'valid': !obj:pylearn2.datasets.tfd.TFD {
                    which_set: 'valid',
                    one_hot: 1,
                    fold: %(fold)i,
                    axes: ['c', 0, 1, 'b'],
                    preprocessor: !obj:pylearn2.datasets.preprocessing.GlobalContrastNormalization {}
                },
                'test': !obj:pylearn2.datasets.tfd.TFD {
                    which_set: 'test',
                    one_hot: 1,
                    fold: %(fold)i,
                    axes: ['c', 0, 1, 'b'],
                    preprocessor: !obj:pylearn2.datasets.preprocessing.GlobalContrastNormalization {}
                },
            },
        cost: !obj:pylearn2.costs.cost.MethodCost {
                method: 'cost_from_X',
                supervised: 1
        },
        termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased {
            channel_name: "valid_y_misclass",
            prop_decrease: 0.,
            N: 100
        },
        update_callbacks: !obj:pylearn2.training_algorithms.sgd.ExponentialDecay {
            decay_factor: %(exp_decay)f,
            min_lr: %(exp_dc_min)f
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "%(save_path)sbest.pkl"
        },
        !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: %(momentum_start)i,
            saturate: %(momentum_saturate)i,
            final_momentum: %(final_momentum)f
        }, !obj:pylearn2.training_algorithms.sgd.LinearDecayOverEpoch {
            start: %(lr_decay_start)i,
            saturate: %(lr_deccay_saturate)i,
            decay_factor: %(lr_decay_factor)f
        }
    ],
    save_path: "%(save_path)slast.pkl",
    save_freq: 1
}
"""

def experiment(state, channel):

    # udate path
    if channel is None:
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path += ''.join(alphabet[:5])

    # load and save yaml
    yaml_string = state.yaml_string % (state)

    with open(state.save_path + '_model.yaml', 'w') as fp:
        fp.write(yaml_string)

    if state.db == 'SVHN':
        # transfer data to tmp
        path = '/RQexec/mirzameh/data/SVHN/h5/'
        tmp_path = '/tmp/data/SVHN/h5/'
        train_f = 'splited_train_32x32.h5'
        valid_f = 'valid_32x32.h5'
        test_f = 'test_32x32.h5'
        if any([not os.path.isfile(path + train_f), os.path.isfile(path + valid_f), os.path.isfile(path + test_f)]):
            try:
                os.mkdir('/tmp/')
            except OSError:
                pass
            try:
                os.mkdir('/tmp/data/')
            except OSError:
                pass
            try:
                os.mkdir('/tmp/data/SVHN/')
            except OSError:
                pass
            try:
                os.mkdir(tmp_path)
            except OSError:
                pass
            shutil.copy(path + train_f, tmp_path + train_f)
            shutil.copy(path + valid_f, tmp_path + valid_f)
            shutil.copy(path + test_f, tmp_path + test_f)

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)
    train_obj.main_loop()

    ext = get_best_params_ext(train_obj.extensions)
    if ext != None:
        state.valid_score = float(ext.best_params['valid_y_misclass'])
        try:
            state.test_score = float(ext.best_params['test_y_misclass'])
        except KeyError:
            state.test_score = -1.

    print "Best valid: {}, best test: {}".format(state.valid_score, state.test_score)
    return channel.COMPLETE

def get_best_params_ext(extensions):
    from noisy_encoder.utils.best_params import MonitorBasedBest
    for ext in extensions:
        if isinstance(ext, MonitorBasedBest):
            return ext

def svhn_experiment():
    state = DD()
    state.yaml_string = train_yaml

    state.data_path = '/data/lisatmp2/mirzamom/data/SVHN/'
    state.num_channels_1 = 128
    state.num_channels_2 = 256
    state.num_channels_3 = 256
    state.max_kernel_norm_1 = 1.2
    state.max_kernel_norm_2 = 2.2
    state.max_kernel_norm_3 = 2.2
    state.learning_rate = 0.5
    state.W_lr_scale_1 = 0.01
    state.W_lr_scale_2 = 0.01
    state.W_lr_scale_3 = 0.01
    state.b_lr_scale_1 = 0.01
    state.b_lr_scale_2 = 0.01
    state.b_lr_scale_3 = 0.01
    state.lr_decay_start = 1
    state.lr_deccay_saturate = 150
    state.lr_decay_factor = 0.001
    state.momentum_start = 1
    state.momentum_saturate = 100
    state.final_momentum = 0.7
    state.max_epochs = 1000
    state.termination_paitence = 100
    state.best_save_path = "/data/lisatmp2/mirzamom/results/ian/3/best.pkl"
    state.save_path = "/data/lisatmp2/mirzamom/results/ian/3/last.pkl"

    experiment(state, None)

def cifar10_experiment():
    state = DD()
    state.yaml_string = sp_train_yaml

    state.num_channels_1 = 96
    state.num_channels_2 = 256
    state.num_channels_3 = 256
    state.channel_pool_size_1 = 1
    state.channel_pool_size_2 = 1
    state.channel_pool_size_3 = 1
    state.max_kernel_norm_1 = 0.9
    state.max_kernel_norm_2 = 1.9365
    state.max_kernel_norm_3 = 1.9365
    state.w_lr_scale_1 = 0.05
    state.w_lr_scale_2 = 0.05
    state.w_lr_scale_3 = 0.05
    state.b_lr_scale_1 = 0.05
    state.b_lr_scale_2 = 0.05
    state.b_lr_scale_3 = 0.05
    state.tied_b_1 = 1
    state.tied_b_2 = 1
    state.tied_b_3 = 1
    state.learning_rate = 0.5
    state.lr_decay_start = 1
    state.lr_deccay_saturate = 250
    state.lr_decay_factor = 0.01
    state.init_momentum = 0.5
    state.momentum_start = 10
    state.momentum_saturate = 50
    state.final_momentum = 0.7
    state.max_epochs = 500
    state.termination_paitence = 100
    state.best_save_path = "/tmp/mirzameh/cifar10_temp_best.pkl"
    state.save_path = "/tmp/mirzameh/cifar_10_temp.pkl"

    experiment(state, None)

def tfd_sp_experiment():
    state = DD()
    state.yaml_string = sp_soft_yaml

    state.fold = 4
    state.num_channels_1 = 96
    state.num_channels_2 = 96
    state.max_kernel_norm_1 = 0.9
    state.max_kernel_norm_2 = 1.9365
    state.W_lr_scale_1 = 0.5
    state.W_lr_scale_2 = 0.5
    state.b_lr_scale_1 = 0.5
    state.b_lr_scale_2 = 0.5
    state.dropout_inp = 1.
    state.learning_rate = 0.5
    state.lr_decay_start = 1
    state.lr_deccay_saturate = 250
    state.lr_decay_factor = 0.01
    state.exp_decay = 1.
    state.exp_dc_min = 0.00001
    state.init_momentum = 0.5
    state.momentum_start = 10
    state.momentum_saturate = 50
    state.final_momentum = 0.7
    state.max_epochs = 500
    state.termination_paitence = 100
    state.save_path = "/data/lisatmp2/mirzamom/results/tfd/4/"

    experiment(state, None)

def tfd_experiment():
    state  = DD()
    with open('tfd_lcn.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'tfd'
    state.fold = 4
    state.num_channels_1 = 96
    state.num_channels_2 = 96
    state.max_kernel_norm_1 = 0.9
    state.max_kernel_norm_2 = 1.9365
    state.W_lr_scale_1 = 0.5
    state.W_lr_scale_2 = 0.5
    state.b_lr_scale_1 = 0.5
    state.b_lr_scale_2 = 0.5
    state.dropout_inp = .8
    state.learning_rate = 0.005
    state.exp_decay = 1.00004
    state.exp_dc_min = 0.000001
    state.init_momentum = 0.5
    state.momentum_start = 10
    state.momentum_saturate = 250
    state.final_momentum = .6
    state.max_epochs = 500
    state.termination_paitence = 100
    state.save_path = "/data/lisatmp2/mirzamom/results/tfd/4/"

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'conv trainer')
    parser.add_argument('-t', '--task', choices = ['svhn', 'cifar10', 'tfd'], required = True)
    args = parser.parse_args()

    if args.task == 'svhn':
        svhn_experiment()
    elif args.task == 'cifar10':
        cifar10_experiment()
    elif args.task == 'tfd':
        tfd_experiment()
    else:
        raise ValueError("Wrong task optipns {}".format(args.task))


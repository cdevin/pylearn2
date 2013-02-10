import os, shutil
import argparse

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


def experiment(state, channel):
    # update base yaml config with jobman commands
    yaml_string = state.yaml_string % (state)

    # save .yaml file if it is jobman job
    if channel != None:
        with open('model.yaml', 'w') as fp:
            fp.write(yaml_string)

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
        state.test_score = float(ext.best_params['test_y_misclass'])

    return channel.COMPLETE

def get_best_params_ext(extensions):
    from noisy_encoder.utils.best_params import MonitorBasedBest
    for ext in extensions:
        if isinstance(ext, MonitorBasedBest):
            return ext


def svhn_experiment():
    state = DD()
    state.yaml_string = train_yaml

    state.data_path = '/tmp/data/SVHN/'
    state.num_channels_1 = 32
    state.num_channels_2 = 32
    state.num_channels_3 = 64
    state.max_kernel_norm_1 = 1.5
    state.max_kernel_norm_2 = 2.9
    state.max_kernel_norm_3 = 2.9
    state.learning_rate = 0.05
    state.W_lr_scale_1 = 0.01
    state.W_lr_scale_2 = 0.01
    state.W_lr_scale_3 = 0.01
    state.b_lr_scale_1 = 0.01
    state.b_lr_scale_2 = 0.01
    state.b_lr_scale_3 = 0.01
    state.lr_decay_start = 10
    state.lr_deccay_saturate = 150
    state.lr_decay_factor = 0.001
    state.momentum_start = 10
    state.momentum_saturate = 50
    state.final_momentum = 0.9
    state.max_epochs = 0
    state.termination_paitence = 100
    state.best_save_path = "/tmp/mirzameh/best.pkl"
    state.save_path = "/tmp/mirzameh/mmodel.pkl"

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'conv trainer')
    parser.add_argument('-t', '--task', choices = ['svhn'], required = True)
    args = parser.parse_args()

    if args.task == 'svhn':
        svhn_experiment()
    else:
        raise ValueError("Wrong task optipns {}".format(args.task))


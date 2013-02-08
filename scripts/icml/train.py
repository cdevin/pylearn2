import os
import argparse

from pylearn2.config import yaml_parse
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

train_yaml = """!obj:pylearn2.train.Train {
    dataset: &train !obj:pylearn2.datasets.svhn.SVHN {
        which_set: 'splited_train',
        path: '/data/lisatmp2/mirzamom/data/SVHN/'
    },
    model: !obj:pylearn2.models.mlp.MLP {
        batch_size: 128,
        layers: [
                 !obj:galatea.mlp.ConvLinearC01B {
                     layer_name: 'h0',
                     pad: 0,
                     detector_channels: 128,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: .9,
                 },
                 !obj:galatea.mlp.ConvLinearC01B {
                     layer_name: 'h1',
                     pad: 3,
                     detector_channels: 128,
                     channel_pool_size: 2,
                     kernel_shape: [8, 8],
                     pool_shape: [4, 4],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
                 },
                 !obj:galatea.mlp.ConvLinearC01B {
                     pad: 3,
                     layer_name: 'h2',
                     detector_channels: 256,
                     channel_pool_size: 4,
                     kernel_shape: [5, 5],
                     pool_shape: [2, 2],
                     pool_stride: [2, 2],
                     irange: .005,
                     max_kernel_norm: 1.9365,
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
        learning_rate: .05,
        init_momentum: .5,
        monitoring_dataset:
            {
                #'train' : *train,
                'valid' : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'valid',
                              path: '/data/lisatmp2/mirzamom/data/SVHN/'
                          },
                'test' : !obj:pylearn2.datasets.svhn.SVHN {
                              which_set: 'test',
                              path: '/data/lisatmp2/mirzamom/data/SVHN/'
                          }
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
            decay_factor: 1.00004,
            min_lr: .000001
        }
    },
    extensions: [
        !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
             channel_name: 'valid_y_misclass',
             save_path: "best.pkl"
        },
        !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
            start: 1,
            saturate: 250,
            final_momentum: .7
        }
    ],
    save_path: "each.pkl",
    save_freq: 1
}"""


def experiment(state, channel):
    # update base yaml config with jobman commands
    yaml_string = state.yaml_string % (state)

    # save .yaml file if it is jobman job
    if channel != None:
        with open('model.yaml', 'w') as fp:
            fp.write(yaml_string)

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


def test_experiment():
    state = DD()
    state.yaml_string = train_yaml
    experiment(state, None)

def mnist_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'mnist'
    state.train_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', center: 0, one_hot: 1, start: 0, stop: 50000}"
    state.valid_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', center: 0, one_hot: 1, start: 50000, stop: 60000}"
    state.test_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'test', center: 0, one_hot: 1}"
    state.init_learning_rate = 0.005
    state.init_momentum = 0.5
    state.final_momentum = 0.99
    state.momentum_start = 30
    state.momentum_saturate = 80
    state.max_epochs = 300
    state.batch_size = 200
    # model params
    state.model = 'conv'
    state.conv_layers = [
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [28, 28],
                            'batch_size' : state.batch_size,
                            'num_channels' : 1,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : None,
                            'kernel_shape' : [5, 5],
                            'num_channels' : None,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                {'name' : 'MaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3]}},
                        #{'name' : 'StochasticMaxPool',
                        #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : [3, 3],
                        #'pool_stride' : [2, 2]}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : None,
                            'batch_size' : state.batch_size,
                            'num_channels' : None,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : None,
                            'kernel_shape' : [5, 5],
                            'num_channels' : None,
                            'num_channels_output' : 64,
                            'batch_size' : state.batch_size,
                            'act_enc' : 'rectifier',}},
                 {'name' : 'MaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3]}},
                        #{'name' : 'StochasticMaxPool',
                        #'params' : {'image_shape' : None,
                        #'num_channels' : None,
                        #'pool_shape' : [3, 3],
                        #'pool_stride' : [2, 2]}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : None,
                            'batch_size' : state.batch_size,
                            'num_channels' : None,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}}]

    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [0.0]
    state.mlp_hidden_corruption_levels = [0.5]
    state.mlp_nunits = [1000]
    state.n_outs = 10
    state.bias_init = 0.1
    state.irange = 0.1
    state.random_filters = False

    state.yaml_string = convolution_yaml_string

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'conv trainer')
    parser.add_argument('-t', '--task', choices = ['tfd_conv', 'mnist_conv', 'cifar10_conv'], required = True)
    args = parser.parse_args()

    test_experiment()

    if args.task == 'tfd_conv':
        tfd_conv_experiment()
    elif args.task == 'mnist_conv':
        mnist_conv_experiment()
    elif args.task == 'cifar10_conv':
        cifar10_conv_experiment()
    else:
        raise ValueError("Wrong task optipns {}".format(args.task))


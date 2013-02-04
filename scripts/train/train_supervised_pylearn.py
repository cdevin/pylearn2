import os
import argparse

from pylearn2.config import yaml_parse
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

# convolution
convolution_yaml_string = """
!obj:pylearn2.train.Train {
    dataset: &train %(train_set)s,
    model: !obj:noisy_encoder.models.conv.LeNetLearner {
        conv_layers: %(conv_layers)s,
        batch_size : %(batch_size)i,
        mlp_act : %(mlp_act)s,
        mlp_input_corruption_levels : %(mlp_input_corruption_levels)s,
        mlp_hidden_corruption_levels : %(mlp_hidden_corruption_levels)s,
        mlp_nunits : %(mlp_nunits)s,
        n_outs : %(n_outs)i,
        irange : %(irange)f,
        bias_init : %(bias_init)f,
        random_filters : %(random_filters)s,
    },
    algorithm: !obj:pylearn2.training_algorithms.sgd.SGD {
        batch_size : %(batch_size)i,
        learning_rate : %(init_learning_rate)f,
        init_momentum : %(init_momentum)f,
        monitoring_dataset : {
            valid : %(valid_set)s,
            test : %(test_set)s},
        cost : !obj:pylearn2.costs.cost.SumOfCosts { costs : [
                !obj:pylearn2.costs.cost.MethodCost {
                    method : 'cost',
                    supervised : 1}]
        },
        termination_criterion: !obj:pylearn2.termination_criteria.And {
            criteria : [!obj:pylearn2.termination_criteria.EpochCounter {
            max_epochs : %(max_epochs)i
            },
            !obj:pylearn2.termination_criteria.MonitorBased {
                channel_name: "valid_misclass",
                prop_decrease: 0.001,
                N: 20}
            ]
        },
    },
    extensions:
        [ !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
            channel_name: 'valid_misclass',
            save_path: "convolutional_network_best.pkl"
        }, !obj:noisy_encoder.utils.best_params.MonitorBasedBest {
            channel_name: 'valid_misclass',
            save_channel_names: ['valid_misclass', 'test_misclass']
        }, !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                start: %(momentum_start)i,
                saturate: %(momentum_saturate)i,
                final_momentum: %(final_momentum)f
        }
        ]
    }
"""

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
        state.valid_score = float(ext.best_params['valid_misclass'])
        state.test_score = float(ext.best_params['test_misclass'])

    return channel.COMPLETE

def get_best_params_ext(extensions):
    from noisy_encoder.utils.best_params import MonitorBasedBest
    for ext in extensions:
        if isinstance(ext, MonitorBasedBest):
            return ext

def mnist_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'mnist'
    state.train_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', center: 1, one_hot: 1, start: 0, stop: 50000}"
    state.valid_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', center: 1, one_hot: 1, start: 50000, stop: 60000}"
    state.test_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'test', center: 1, one_hot: 1}"
    state.init_learning_rate = 0.005
    state.init_momentum = 0.5
    state.final_momentum = 0.99
    state.momentum_start = 30
    state.momentum_saturate = 80
    state.max_epochs = 300
    state.batch_size = 100
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
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3],
                        'pool_stride' : [2, 2]}},
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
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3],
                        'pool_stride' : [2, 2]}},
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

def tfd_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'tfd'
    state.fold = 0
    #state.train_set = "!obj:pylearn2.datasets.tfd.TFD {which_set: 'train', one_hot: 1}"
    #state.valid_set = "!obj:pylearn2.datasets.tfd.TFD {which_set: 'valid', one_hot: 1}"
    #state.test_set = "!obj:pylearn2.datasets.tfd.TFD {which_set: 'test', one_hot: 1}"
    state.train_set = "!pkl: " + os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/train.pkl".format(state.fold))
    state.valid_set = "!pkl: " + os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/test.pkl".format(state.fold))
    state.test_set = "!pkl: " + os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/valid.pkl".format(state.fold))
    state.init_learning_rate = 0.0001
    state.init_momentum = 0.5
    state.max_epochs = 0
    state.batch_size = 1
    state.save_name = os.path.join(RESULT_PATH, "naenc/tfd/tfd_gpu.pkl")
    state.coeffs = {'w_l1' : 0.0, 'w_l2' : 0e-5}
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
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3],
                        'pool_stride' : [2, 2]}},
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
                {'name' : 'StochasticMaxPool',
                    'params' : {'image_shape' : None,
                        'num_channels' : None,
                        'pool_shape' : [3, 3],
                        'pool_stride' : [2, 2]}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : None,
                            'batch_size' : state.batch_size,
                            'num_channels' :None ,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75
                            }}]
    state.mlp_act = "rectifier"
    state.mlp_input_corruption_levels = [0.0, 0.0]
    state.mlp_hidden_corruption_levels = [0.5, 0.5]
    state.mlp_nunits = [1200]
    state.n_outs = 7
    state.bias_init = 0.1
    state.irange = 0.1
    state.random_filters = False

    state.yaml_string = convolution_yaml_string

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'noisy AE trainer')
    parser.add_argument('-t', '--task', choices = ['tfd_conv', 'mnist_conv'], required = True)
    args = parser.parse_args()

    if args.task == 'tfd_conv':
        tfd_conv_experiment()
    if args.task == 'mnist_conv':
        mnist_conv_experiment()
    else:
        raise ValueError("Wrong task optipns {}".format(args.task))


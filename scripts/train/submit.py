import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train_supervised_pylearn import convolution_yaml_string
from noisy_encoder.scripts.train_supervised_pylearn import experiment
from my_utils.config import get_experiment_path



def mnist_conv_experiment():

    state = DD()

    # train params
    state.dataset = 'mnist'
    state.train_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', one_hot: 1, start: 0, stop: 50000}"
    state.valid_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'train', one_hot: 1, start: 50000, stop: 60000}"
    state.test_set = "!obj:pylearn2.datasets.mnist.MNIST {which_set: 'test', one_hot: 1}"
    state.init_learning_rate = 0.005
    state.init_momentum = 0.5
    state.final_momentum = 0.99
    state.momentum_start = 30
    state.momentum_saturate = 80
    state.max_epochs = 0
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

    ind = 0
    TABLE_NAME = "tfd_random_conv_arch"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.005, 0.0005]:
        state.init_learning_rate = lr
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

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

    ind = 0
    TABLE_NAME = "tfd_random_conv_arch"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.005, 0.0005]:
        state.init_learning_rate = lr
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['tfd_conv', 'mnist_conv'])
    args = parser.parse_args()

    if args.task == 'tfd_conv':
        tfd_conv()
    elif args.task == 'mnist_conv':
        mnist_conv()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



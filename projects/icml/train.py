import os, shutil
import argparse
import numpy
from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import serial
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

def experiment(state, channel):

    # udate path
    if channel is None:
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path += ''.join(alphabet[:7]) + '_'

    # load and save yaml
    yaml_string = state.yaml_string % (state)
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    if state.db == 'SVHN':
        # transfer data to tmp
        orig_path = state.orig_path + 'h5/'
        tmp_path = state.data_path + 'h5/'
        train_f = 'splitted_train_32x32.h5'
        valid_f = 'valid_32x32.h5'
        test_f = 'test_32x32.h5'
        if any([not os.path.isfile(tmp_path + train_f), not os.path.isfile(tmp_path + valid_f), not os.path.isfile(tmp_path + test_f)]):
            print "Moving data to local tmp"
            os.makedirs(tmp_path)

            shutil.copy(orig_path + train_f, tmp_path)
            shutil.copy(orig_path + valid_f, tmp_path)
            shutil.copy(orig_path + test_f, tmp_path)

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


def experiment_finetune(state, channel, pre_trained):

    # udate path
    if channel is None:
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path += ''.join(alphabet[:7]) + '_'

    # load and save yaml
    yaml_string = state.yaml_string % (state)
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)

    # load the pre-trained models
    model = serial.load(pre_trained)
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
    with open('exp/svhn_transform.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN'

    state.orig_path = preprocess('${PYLEARN2_DATA_CUSTOM}/SVHN/icpr/')
    state.data_path = preprocess('${PYLEARN2_DATA_TMP}/SVHN/icpr/')
    state.num_channels_0 = 32
    state.num_channels_1 = 32
    state.num_channels_2 = 32
    state.num_units_0 = 60
    state.num_units_1 = 40
    state.learning_rate = 0.5
    state.decay_factor = 0.01
    #state.lr_min_lr = 0.00001
    #state.lr_decay_factor = 1.00004
    #state.momentum_start = 1
    #state.momentum_saturate = 100
    #state.final_momentum = 0.65
    #state.termination_paitence = 100
    state.irange = 0.005
    state.save_path = preprocess('${PYLEARN2_EXP_RESULTS}/svhn/sot/')
    state.file_type = 'joblib'

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

def cifar10_experiment():
    state = DD()
    with open('exp/cifar_sp.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'CIFAR10'

    state.learning_rate = 0.5
    state.save_path = preprocess('${PYLEARN2_EXP_RESULTS}/cifar10/sp/')

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'conv trainer')
    parser.add_argument('-t', '--task', choices = ['svhn', 'cifar10', 'tfd', 'svhn_size'], required = True)
    args = parser.parse_args()

    if args.task == 'svhn':
        svhn_experiment()
    elif args.task == 'cifar10':
        cifar10_experiment()
    elif args.task == 'tfd':
        tfd_experiment()
    elif args.task == 'svhn_size':
        svhn_train_size_experiment()
    else:
        raise ValueError("W rong task optipns {}".format(args.task))


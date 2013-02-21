import os, shutil
import argparse
import numpy
from pylearn2.config import yaml_parse
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

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

    if state.db == 'SVHN_briee':
        # transfer data to tmp
        path = '/RQexec/mirzameh/data/SVHN/h5/'
        tmp_path = '/tmp/data/SVHN/h5/'
        train_f = 'splitted_train_32x32.h5'
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
    with open('exp/svhn.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN'

    state.data_path = '/Tmp/mirzamom/data/SVHN/'
    #state.data_path = '/tmp/data/SVHN/'
    state.num_channels_0 = 96
    state.num_channels_1 = 128
    state.num_channels_2 = 128
    state.layer_ndim = 500
    #state.channel_pool_size_1 = 2
    #state.channel_pool_size_2 = 2
    #state.channel_pool_size_3 = 4
    #state.pool_shape_1 = '[4, 4]'
    #state.pool_shape_2 = '[4, 4]'
    #state.pool_shape_3 = '[2, 2]'
    #state.kernel_shape_1 = '[8, 8]'
    #state.kernel_shape_2 = '[8, 8]'
    #state.kernel_shape_3 = '[5, 5]'
    #state.pool_stride_1 = '[2, 2]'
    #state.pool_stride_2 = '[2, 2]'
    #state.pool_stride_3 = '[2, 2]'
    #state.max_kernel_norm_1 = 1.2
    #state.max_kernel_norm_2 = 2.2
    #state.max_kernel_norm_3 = 2.2
    state.learning_rate = 0.1
    #state.lr_min_lr = 0.00001
    #state.lr_decay_factor = 1.00004
    #state.momentum_start = 1
    #state.momentum_saturate = 100
    #state.final_momentum = 0.65
    #state.termination_paitence = 100
    state.save_path = "/data/lisatmp2/mirzamom/results/svhn_linearpool/sot/"

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

def svhn_train_size_experiment():
    state = DD()
    #with open('lp_svhn.yaml') as ymtmp:
    #with open('rec_svhn.yaml') as ymtmp:
    #with open('lp_nodroput_svhn.yaml') as ymtmp:
    #with open('rec_nodropout_svhn.yaml') as ymtmp:
    #with open('lp_svhn_60k.yaml') as ymtmp:
    #with open('rec_svhn_60k.yaml') as ymtmp:
    #with open('exp/lp_nodroput_svhn_60k.yaml') as ymtmp:
    #with open('exp/rec_nodropout_svhn_60k.yaml') as ymtmp:
    #with open('exp/lp_svhn_6k.yaml') as ymtmp:
    #with open('exp/rec_svhn.yaml') as ymtmp:
    #with open('exp/lp_nodroput_svhn.yaml') as ymtmp:
    #with open('exp/lp_nodroput_svhn_60k.yaml') as ymtmp:
    #with open('exp/rec_nodropout_svhn.yaml') as ymtmp:
    #with open('exp/lp_nodroput_svhn_6k.yaml') as ymtmp:
    #with open('exp/rec_svhn_6k.yaml') as ymtmp:
    #with open('exp/rec_nodropout_svhn_6k.yaml') as ymtmp:
    with open('exp/rec_svhn_100k.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN_briee'
    #state.data_path = '/data/lisatmp/mirzamom/data/SVHN/600k/'
    state.data_path = '/tmp/data/SVHN/'
    state.learning_rate = 0.05
    state.lr_decay_factor = 1.000004
    state.lr_min_lr = .000001
    state.momentum_start = 1
    state.momentum_saturate = 50
    state.final_momentum = 0.7
    state.max_epochs = 300
    state.save_path = "/RQexec/mirzameh/results/svhn/100k/"
    #state.save_path = "/data/lisatmp2/mirzamom/results/svhn_train_size_test/600k/"

    experiment(state , None)

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
        raise ValueError("Wrong task optipns {}".format(args.task))


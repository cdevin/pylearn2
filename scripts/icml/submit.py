import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.icml.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def svhn_ian():
    state = DD()
    state.yaml_string = train_yaml

    state.data_path = '/tmp/data/SVHN/'
    state.num_channels_1 = 32
    state.num_channels_2 = 32
    state.num_channels_3 = 64
    state.max_kernel_norm_1 = 1.5
    state.max_kernel_norm_2 = 2.5
    state.max_kernel_norm_3 = 2.5
    state.learning_rate = 0.05
    state.W_lr_scale_1 = 0.01
    state.W_lr_scale_2 = 0.01
    state.W_lr_scale_3 = 0.01
    state.b_lr_scale_1 = 0.01
    state.b_lr_scale_2 = 0.01
    state.b_lr_scale_3 = 0.01
    state.lr_decay_start = 10
    state.lr_deccay_saturate = 110
    state.lr_decay_factor = 0.001
    state.momentum_start = 5
    state.momentum_saturate = 50
    state.final_momentum = 0.7
    state.max_epochs = 500
    state.termination_paitence = 100
    state.best_save_path = "best.pkl"
    state.save_path = "current.pkl"

    ind = 0
    TABLE_NAME = "ian_svhn"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.5, 0.1, 0.05, 0.005]:
        for ch1, ch2, ch3 in zip([128, 128], [128, 256], [256, 512]):
            state.learning_rate = lr
            state.num_channels_1 = ch1
            state.num_channels_2 = ch2
            state.num_channels_3 = ch3
            sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def cifar10():

    state = DD()
    state.yaml_string = sp_train_yaml

    state.num_channels_1 = 96
    state.num_channels_2 = 256
    state.num_channels_3 = 256
    state.channel_pool_size_1 = 2
    state.channel_pool_size_2 = 2
    state.channel_pool_size_3 = 4
    state.max_kernel_norm_1 = 0.9
    state.max_kernel_norm_2 = 1.9365
    state.max_kernel_norm_3 = 1.9365
    state.W_lr_scale_1 = 1.
    state.W_lr_scale_2 = 1.
    state.W_lr_scale_3 = 1.
    state.b_lr_scale_1 = 1.
    state.b_lr_scale_2 = 1.
    state.b_lr_scale_3 = 1.
    state.tied_b_1 = 0
    state.tied_b_2 = 0
    state.tied_b_3 = 0
    state.learning_rate = 0.05
    state.lr_decay_start = 10
    state.lr_deccay_saturate = 150
    state.lr_decay_factor = 0.0001
    state.init_momentum = 0.5
    state.momentum_start = 10
    state.momentum_saturate = 50
    state.final_momentum = 0.7
    state.max_epochs = 500
    state.termination_paitence = 100
    state.best_save_path = "best.pkl"
    state.save_path = "last.pkl"

    ind = 0
    TABLE_NAME = "ian_cifar10"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.5, 0.1, 0.05, 0.005]:
        for dc in [0.001, 0.0001]:
            state.learning_rate = lr
            state.lr_decay_factor = dc
            sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def svhn():

    state = DD()
    with open('exp/svhn.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN'
    state.data_path = preprocess('${PYLEARN2_DATA_TMP}/SVHN/channel/')
    state.num_channels_0 = 64
    state.num_channels_1 = 128
    state.num_channels_2 = 192
    state.num_units = 240
    state.learning_rate = 0.1
    state.save_path = "./"

    ind = 0
    TABLE_NAME = "svhn_full"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.5, 0.1, 0.05, 0.005]:
        state.learning_rate = lr
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


def cifar10_sp():

    state = DD()
    with open('exp/cifar_sp.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'CIFAR10'

    state.learning_rate = 0.1
    state.save_path = "./"


    ind = 0
    TABLE_NAME = "cifar10_sp22"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.5, 0.1, 0.05, 0.01]:
        state.learning_rate = lr
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['svhn_ian', 'cifar10', 'cifar10_sp', 'svhn'])
    args = parser.parse_args()

    if args.task == 'svhn_ian':
        svhn_ian()
    elif args.task == 'cifar10':
        cifar10()
    elif args.task == 'cifar10_sp':
        cifar10_sp()
    elif args.task == 'svhn':
        svhn()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



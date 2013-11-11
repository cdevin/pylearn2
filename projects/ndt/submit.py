import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def cifar(submit = False):
    state =  DD()
    with open('exp/cifar_aug.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'cifar'
    state.learning_rate = 0.1
    state.lr_decay = 0.001
    if submit:
        state.save_path = './'
        TABLE_NAME = "tree_cifar_aug"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/tree/cifar10_aug/")

    ind = 0
    for lr in [1., 0.5, 0.2]:
        for lr_decay in [0.1, 0.01, 0.001]:
            state.lr = lr
            state.lr_decay = lr_decay
            if submit:
                sql.insert_job(experiment, flatten(state), db)
            else:
                experiment(state, None)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def cifar_bin(submit = False):
    state =  DD()
    with open('exp/cifar_bin.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'cifar'
    state.learning_rate = 0.1
    state.lr_decay = 0.001
    if submit:
        state.save_path = '. /'
        TABLE_NAME = "tree_cifar_bin"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/tree/cifar10_aug/")

    ind = 0
    for lr in [1., 0.5, 0.2]:
        for lr_decay in [0.1, 0.01, 0.001]:
            state.lr = lr
            state.lr_decay = lr_decay
            if submit:
                sql.insert_job(experiment, flatten(state), db)
            else:
                experiment(state, None)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted" .format(ind)



if __name__ == "__main__":

    parser  =  argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['cifar', 'cifar_bin'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'cifar':
        cifar(args.submit)
    elif args.task == 'cifar_bin':
        cifar_bin(args.submit)
    else:
        raise ValueErr or("Wrong task optipns {}".fromat(args.task))



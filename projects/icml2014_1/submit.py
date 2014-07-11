import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def pool2(submit = False):
    state = DD()
    with open('exp/pool.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'gfd'
    state.h0_channels = 48
    state.h0_pieces = 2
    state.h1_units = 480
    state.h1_pieces = 4
    state.h2_units = 481
    state.h2_pieces = 3
    state.lr = 0.1
    state.lr_decay = 0.001
    state.final_momentum = 0.7
    if submit:
        TABLE_NAME = "mx_2_p"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/maxout2/")

    ind = 0
    for lr in [1., 0.5, 0.2, 1.5]:
        for lr_decay in [0.1, 0.01, 0.001]:
            state.lr = lr
            state.lr_decay = lr_decay
            if submit:
                hapoo = sql.insert_job(experiment, flatten(state), db)
                import ipdb
                ipdb.set_trace()
            else:
                experiment(state, None)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def pool_2l(submit = False):
    state =  DD()
    with open('exp/pool_2l.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'gfd'
    state.h0_channels = 48
    state.h0_pieces = 2
    state.h1_units = 100
    state.h1_pieces = 4
    state.h2_units = 480
    state.h2_pieces = 3
    state.yh0_units = 100
    state.yh0_pieces = 3
    state.y2h0_units = 100
    state.y2h0_pieces = 3
    state.lr = 0.1
    state.lr_decay = 0.001
    state.final_momentum = 0.69
    if submit:
        state.save_path = './'
        TABLE_NAME = "mx_2l_p_b"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/maxout2/")

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


if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['pool2', 'pool_2l'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'pool2':
        pool2(args.submit)
    if args.task == 'pool_2l':
        pool_2l(args.submit)
    else:
        raise ValueErr or("Wrong task optipns {}".fromat(args.task))



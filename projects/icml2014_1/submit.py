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

    state.db = 'mnist'
    state.h0_channels = 48
    state.h0_pieces = 3
    state.h1_units = 400
    state.h1_pieces = 4
    state.lr = 0.1
    state.lr_decay = 0.001
    state.final_momentum = 0.7
    state.save_path = './'
    #state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/maxout2/")

    ind = 0
    TABLE_NAME = "mx_2"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.1, 0.5]:
        for lr_decay in [0.01, 0.001]:
            state.lr = 0.1
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
    parser.add_argument('-t', '--task', choices = ['pool2'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'pool2':
        pool2(args.submit)
    else:
        raise ValueErr or("Wrong task optipns {}".fromat(args.task))



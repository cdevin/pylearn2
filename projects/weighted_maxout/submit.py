import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def kl():
    state = DD()
    with open('exp/kl.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'mnist'
    state.h0_units = 1000
    state.h0_pieces = 10
    state.h1_units = 1000
    state.h1_pieces = 10
    state.sparsity_ratio_0 = 0.1
    state.sparsity_ratio_1 = 0.1
    state.learning_rate = 0.1
    state.decay_factor = 0.01
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/rec/")

    ind = 0
    TABLE_NAME = "gate_kl"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        for dec in [0.001]:
            state.learning_rate = lr
            state.decay_factor = dec
            experiment(state, None)
            #sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['kl'])
    args = parser.parse_args()

    if args.task == 'kl':
        kl()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



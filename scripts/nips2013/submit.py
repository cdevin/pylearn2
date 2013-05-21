import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def norb():
    state = DD()
    with open('exp/norb_9D2.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'norb'
    state.learning_rate = 2.
    state.decay_factor = 0.02
    state.lr_saturate = 868
    state.m_saturate = 2
    state.save_path = './'


    ind = 0
    TABLE_NAME = "pdbm_norb"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.2]:
        for dec in [0.1, 0.01]:
            state.learning_rate = lr
            state.decay_factor = dec
            #experiment(state, None)
            sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['norb'])
    args = parser.parse_args()

    if args.task == 'norb':
        norb()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



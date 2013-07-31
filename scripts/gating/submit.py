import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def rec():
    state = DD()
    with open('exp/rec.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'ocr'
    state.layer0_dim = 3000
    state.layer1_dim = 4000
    state.niter = 5
    state.noise = 1
    state.learning_rate = 2.
    state.decay_factor = 0.066484
    state.lr_saturate = 200
    state.m_saturate = 2
    state.final_momentum = 0.802294
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/rec/")

    ind = 0
    TABLE_NAME = "gate_rec"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [10, 15, 5]:
        for dec in [0.01, 1.]:
            state.learning_rate = lr
            state.decay_factor = dec
            experiment(state, None)
            #sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['rec])
    args = parser.parse_args()

    if args.task == 'rec':
        rec()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def frame():
    state = DD()
    with open('exp/frame_conv.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.01
    state.h0_num_channels = 64
    state.h1_num_channels = 128
    state.last_ndim = 400
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_frame_seq"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        state.learning_rate = lr
        experiment(state, None)
        #sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def crf():
    state = DD()
    with open('exp/crf_conv.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.1
    state.h0_num_channels = 16
    state.h1_num_channels = 16
    state.last_ndim = 100
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_crf_seq"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.0001]:
        state.learning_rate = lr
        experiment(state, None)
        #sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['frame', 'crf'])
    args = parser.parse_args()

    if args.task == 'frame':
        frame()
    elif args.task == 'crf':
        crf()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



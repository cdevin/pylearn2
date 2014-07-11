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

def framep():
    state  = DD()
    with open('exp/frame_tfd.yaml') as ymtmp:
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
    for lr in [0.05]:
        state.learning_rate = lr
        experiment(state, None)
        #sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def seq3d():
    state = DD()
    with open('exp/frame_3dconv.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.01
    state.h0_num_channels = 64
    state.h1_num_channels = 128
    state.last_ndim = 400
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_frame_3dseq"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        state.learning_rate = lr
        experiment(state, None)
        #sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print  "{} jobs submitted".format(ind)

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

def tfd():
    state = DD()
    with open('exp/tfd.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.5
    state.lr_de_fac = 0.001
    state.h0_ch = 64
    state.h1_ch = 64
    state.h2_ch = 128
    state.h3_units = 300
    state.fold = 0
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_tfd"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.5, 0.1, 1.]:
        for dec in [0.001, 0.001]:
            state.lr_de_fac = dec
        state.learning_rate = lr
        experiment(state, None)
        #sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def google():
    state = DD()
    with open('exp/google_tfd.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.5
    state.lr_de_fac = 0.001
    state.h0_ch = 64
    state.h1_ch = 64
    state.h2_ch = 128
    state.h3_units = 240
    state.fold = 0
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_google"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.5]:
        for dec in [0.001]:
            state.lr_de_fac = dec
            state.learning_rate = lr
            experiment(state, None)
            #sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def crfp():
    state = DD()
    with open('exp/crf_conv_p.yaml') as ymtmp:
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

def feat():
    state = DD()
    with open('exp/feat.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.5
    state.lr_de_fac = 0.001
    state.h0_units = 500
    state.h1_units = 500
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_tfd"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        for dec in [0.001]:
            state.lr_de_fac = dec
            state.learning_rate = lr
            experiment(state, None)
            #sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def combo():
    state = DD()
    with open('exp/combo.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'afew'
    state.learning_rate = 0.5
    state.lr_de_fac = 0.001
    state.h0_ch = 64
    state.h1_ch = 64
    state.h2_ch = 128
    state.h3_units = 240
    state.fold = 0
    #state.save_path = './'
    state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/challenge/frame/")

    ind = 0
    TABLE_NAME = "challenge_google"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.2]:
        for dec in [0.001]:
            state.lr_de_fac = dec
            state.learning_rate = lr
            experiment(state, None)
            #sql.insert_job(experiment, flatten(state), db)
            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['frame', 'crf', 'crfp', 'seq3d', 'tfd', 'feat', 'google', 'framep', 'combo'])
    args = parser.parse_args()

    if args.task == 'frame':
        frame()
    elif args.task == 'crf':
        crf()
    elif args.task == 'crfp':
        crfp()
    elif args.task == 'seq3d':
        seq3d()
    elif args.task == 'tfd':
        tfd()
    elif args.task == 'feat':
        feat()
    elif args.task == 'google':
        google()
    elif args.task == 'framep':
        framep()
    elif args.task == 'combo':
        combo()
    else:
         raise ValueError("Wrong task optipns {}".fromat(args.task))



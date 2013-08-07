import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def softmax(submit = False):
    state = DD()
    with open('exp/sigmoid.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'mnist'
    state.h0_units = 1000
    state.h0_pieces = 10
    state.h1_units = 1000
    state.h1_pieces = 10
    sate.selection_type = 'one_hot'
    state.sparsity_ratio_0 = 0.1
    state.sparsity_ratio_1 = 0.1
    state.sparsity_momentum_0 = 0.9
    state.sparsity_momentum_1 = 0.9
    state.sparsity_penalty = 0.1
    state.learning_rate = 0.1
    state.decay_factor = 0.001
    state.save_path = './'
    #state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/sig/")

    ind = 0
    TABLE_NAME = "gate_kl_sm"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1, 0.01, 0.5]:
        for s0, s1 in zip ([0.1, 0.01], [0.1, 0.01]):
            for sp in [0.1, 0.01, 0.001, 1., 10]:
                state.learning_rate = lr
                state.sparsity_ratio_0 = s0
                state.sparsity_ratio_1 = s1
                state.sparsity_penalty = sp
                if submit:
                    sql.insert_job(experiment, flatten(state), db)
                else:
                    experiment(state, None)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def kl_sig(submit = False):
    state = DD()
    with open('exp/kl_sigmoid.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'mnist'
    state.h0_units = 1000
    state.h0_pieces = 10
    state.h1_units = 1000
    state.h1_pieces = 10
    state.sparsity_ratio_0 = 0.1
    state.sparsity_ratio_1 = 0.1
    state.sparsity_momentum_0 = 0.9
    state.sparsity_momentum_1 = 0.9
    state.sparsity_penalty = 0.1
    state.learning_rate = 0.1
    state.decay_factor = 0.001
    state.save_path = './'
    #state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/sig/")

    ind = 0
    TABLE_NAME = "gate_kl_sig3"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1, 0.01, 0.5]:
        for s0, s1 in zip ([0.1, 0.01], [0.1, 0.01]):
            for sp in [0.1, 0.01, 0.001, 1., 10]:
                state.learning_rate = lr
                state.sparsity_ratio_0 = s0
                state.sparsity_ratio_1 = s1
                state.sparsity_penalty = sp
                if submit:
                    sql.insert_job(experiment, flatten(state), db)
                else:
                    experiment(state, None)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def normal(submit):
    state  = DD()
    with open('exp/mnist.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'mnist'
    state.learning_rate = 0.1
    state.decay_factor = 0.001
    state.save_path = './'
    state.min_zero = 1
    #state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/sig/")

    ind = 0
    TABLE_NAME = "gate_normal2"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1, 0.01, 0.5, 0.05]:
        state.learning_rate = lr
        if submit:
            sql.insert_job(experiment, flatten(state), db)
        else:
            experiment(state, None)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def stochastic(submit = False):
    state = DD()
    with open('exp/softmax.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'mnist'
    state.h0_units = 1000
    state.h0_pieces = 10
    state.h1_units = 1000
    state.h1_pieces = 10
    state.selection_type = 'stochastic'
    state.sparsity_ratio_0 = 0.1
    state.sparsity_ratio_1 = 0.1
    state.sparsity_momentum_0 = 0.9
    state.sparsity_momentum_1 = 0.9
    state.sparsity_penalty = 0.1
    state.learning_rate = 0.1
    state.decay_factor = 0.001
    state.save_path = './'
    #state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/gating/sig/")

    ind = 0
    TABLE_NAME = "gate_kl_sm"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        for s0, s1 in zip ([0.1], [0.1]):
            for sp in [0.1]:
                state.learning_rate = lr
                state.sparsity_ratio_0 = s0
                state.sparsity_ratio_1 = s1
                state.sparsity_penalty = sp
                if submit:
                    sql.insert_job(experiment, flatten(state), db)
                else:
                    experiment(state, None)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['softmax', 'kl_sig', 'normal', 'stochastic'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'softmax':
        softmax(args.submit)
    elif args.task == 'kl_sig':
        kl_sig(args.submit)
    elif args.task == 'normal':
        normal(args.submit)
    elif args.task == 'stochastic':
        stochastic(args.submit)
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



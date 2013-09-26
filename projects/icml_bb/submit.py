import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
#from noisy_encoder.scripts.icml.train import experiment
from noisy_encoder.scripts.train import experiment, experiment_finetune
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def blackbox_finetune():

    state = DD()
    with open('exp/mlp_2.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    state.n_units_0 = 1000
    state.n_units_1 = 1000
    state.n_units_2 = 1000
    state.n_units_3 = 1000
    state.n_pieces_0 = 8
    state.n_pieces_1 = 8
    state.n_pieces_2 = 8
    state.n_pieces_3 = 8
    state.lr_init = 0.1
    state.lr_saturate = 250
    state.lr_decay_factor = 0.01
    state.save_path = "./"
    state.description = "2 layer standardize preprocessing"


    ind = 0
    TABLE_NAME = "blackbox4"
    #db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        for dec in [0.01]:
            for sat in [250]:
                state.lr_init = lr
                state.lr_decay_factor = dec
                state.lr_saturate = sat
                #experiment(state, None)
                experiment_finetune(state, None, 'last.pkl')
                #sql.insert_job(experiment, flatten(state), db)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def blackbox_cae():

    state = DD()
    with open('exp/cae.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    state.n_units_0 = 1000
    state.n_units_1 = 1000
    state.n_units_2 = 1000
    state.n_units_3 = 1000
    state.contraction_penalty = 0.001
    state.max_epochs = 100
    state.lr_init = 0.01
    state.lr_saturate = 250
    state.lr_decay_factor = 0.01
    state.save_path = "./"
    state.description = "2 layer standardize preprocessing"


    ind = 0
    TABLE_NAME = "blackbox4"
    #db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1]:
        for dec in [0.01]:
            for sat in [250]:
                state.lr_init = lr
                state.lr_decay_factor = dec
                state.lr_saturate = sat
                #experiment(state, None)
                experiment_finetune(state, None, 'last.pkl')
                #sql.insert_job(experiment, flatten(state), db)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def blackbox():

    state = DD()
    with open('exp/max_5.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    state.n_units_0 = 150
    state.n_units_1 = 150
    state.n_units_2 = 150
    state.n_units_3 = 150
    state.n_units_4 = 150
    state.n_pieces_0 = 10
    state.n_pieces_1 = 10
    state.n_pieces_2 = 10
    state.n_pieces_3 = 10
    state.n_pieces_4 = 10
    state.lr_init = 0.1
    state.lr_saturate = 250
    state.lr_decay_factor = 0.001
    state.input_drop = .8
    state.save_path = "./"
    state.description = "4 layer standardize preprocessing"


    ind = 0
    TABLE_NAME = "blackbox5"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    #for seed in [232, 1322, 3409, 94333, 238, 83871, 2992, 7292, 23872, 23872, 28328, 121]:
    for seed in [7292]:
        for lr in [0.4, 0.2, 0.1, 0.01, .15, .25]:
            for dec in [0.1, 0.01]:
                for sat in [250, 100, 50]:
                    state.seed = seed
                    state.lr_init = lr
                    state.lr_decay_factor = dec
                    state.lr_saturate = sat
                    #experiment(state, None)
                    sql.insert_job(experiment, flatten(state), db)
                    ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def blackbox_seed():

    state = DD()
    with open('exp/518.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    #state.n_units_0 = 1000
    #state.n_units_1 = 1000
    #state.n_pieces_0 = 8
    #state.n_pieces_1 = 8
    #state.lr_init = 0.1
    #state.lr_saturate = 250
    #state.lr_decay_factor = 0.001
    state.save_path = "./"
    state.description = "2 layer standardize preprocessing"

    import random

    ind = 0
    TABLE_NAME = "blackbox_seed_3small"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for seed in random.sample(range(10000), 20):
        state.seed = seed
        #state.seed_mlp = ms
        #experiment(state, None)
        sql.insert_job(experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['blackbox', 'bb_s','bb_cae', 'bb_ft'])
    args = parser.parse_args()

    if args.task == 'blackbox':
        blackbox()
    elif args.task == 'bb_s':
        blackbox_seed()
    elif args.task == 'bb_cae':
        blackbox_cae()
    elif args.task == 'bb_ft':
        blackbox_finetune()
    else:
        raise ValueError("W rong task optipns {}".fromat(args.task))



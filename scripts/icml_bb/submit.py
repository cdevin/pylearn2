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
    with open('exp/mlp_3.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    state.n_units_0 = 800
    state.n_units_1 = 800
    state.n_units_2 = 800
    state.n_pieces_0 = 8
    state.n_pieces_1 = 8
    state.n_pieces_2 = 8
    state.lr_init = 0.1
    state.lr_saturate = 250
    state.lr_decay_factor = 0.001
    state.input_drop = 0.8
    state.irange = 0.05
    state.sf_irange = 0.05
    state.save_path = "./"
    state.description = "2 layer standardize preprocessing"


    ind = 0
    TABLE_NAME = "blackbox3"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [1., 0.1, 0.01]:
        for dec in [0.1, 0.01, 0.001]:
            for sat in [250]:
                state.lr_init = lr
                state.lr_decay_factor = dec
                state.lr_saturate = sat
                experiment(state, None)
                sql.insert_job(experiment, flatten(state), db)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def blackbox_seed():

    state = DD()
    with open('exp/mlp_seed.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'blackbox'

    state.n_units_0 = 1000
    state.n_units_1 = 1000
    state.n_pieces_0 = 8
    state.n_pieces_1 = 8
    state.lr_init = 0.1
    state.lr_saturate = 250
    state.lr_decay_factor = 0.001
    state.save_path = "./"
    state.description = "2 layer standardize preprocessing"


    ind = 0
    TABLE_NAME = "blackbox_seed"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    dsl = [73, 2383, 29080, 232, 1210, 7734, 3232, 8349, 48303, 8833]
    msl = [32, 89, 882, 823, 902, 9083 ,834, 97430, 932, 9922]
    for ds, ms in zip(dsl, msl):
        state.seed = ds
        state.seed_mlp = ms
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



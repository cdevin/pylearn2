import os
import argparse, fnmatch
import numpy as np
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()


def tfd(submit = False, make = False):
    state = DD()
    with open('exp/tfd.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'tfd'
    state.nvis = 200
    state.g_dim0 = 8000
    state.g_dim1 = 8000
    state.d_dim0 = 2000
    state.d_dim1 = 2000
    state.d_piece0 = 3
    state.d_piece1 = 3
    state.batch_size = 100
    state.lr = 0.1
    state.term_dec = 0.1
    state.momentun_saturate = 100
    state.final_momentum = 0.7
    state.lr_sat = 100
    state.decay = 0.1
    num_exp = 30
    if submit:
        TABLE_NAME = "adv_tfd"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/adv/tfd/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.g_h0_col_norm = rng.uniform(1., 3.)
        state.g_h1_col_norm = rng.uniform(1., 3.)
        state.g_y_col_norm = rng.uniform(1., 3.)
        state.d_h0_col_norm = rng.uniform(1., 3.)
        state.d_h1_col_norm = rng.uniform(1., 3.)
        state.d_y_col_norm = rng.uniform(1., 5.)

        state.nvis = rng.randint(50, 2000)
        state.g_dim0 = rng.randint(2000, 12000)
        state.g_dim2 = rng.randint(2000, 12000)
        state.d_dim0 = rng.randint(500, 5000)
        state.d_dim1 = rng.randint(500, 5000)
        state.d_pieces0 = rng.randint(2, 5)
        state.d_pieces1 = rng.randint(2, 5)

        state.lr = 10. ** rng.uniform(1., -3)
        state.term_dec = 10 ** rng.uniform(-1, -3)
        state.momentum_saturate = rng.randint(2, 200)
        state.final_momentum = rng.uniform(.5, .7)
        #state.lr_sat =rng.randint(50, 200)
        state.lr_decay = 10. ** rng.uniform(-3, -1)

        def random_init_string():
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string()
        state.h1_init = random_init_string()
        state.h2_init = random_init_string()
        state.h3_init = random_init_string()
        if rng.randint(2):
            state.y_init = "sparse_init: 0"
        else:
            state.y_init = random_init_string()

        if make:
            state.save_path = os.path.join(PATH, str(i)) + '/'
            if not os.path.isdir(state.save_path):
                os.mkdir(state.save_path)
            yaml = state.yaml_string % (state)
            with open(os.path.join(state.save_path, 'model.yaml'), 'w') as fp:
                fp.write(yaml)
        else:
            if submit:
                sql.insert_job(experiment, flatten(state), db)
            else:
                experiment(state, None)

    if not make:
        db.createView(TABLE_NAME + '_view')

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['tfd'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    parser.add_argument('-m', '--make', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'tfd':
        tfd(args.submit)
    else:
        raise ValueError or("Wrong task optipns {}".fromat(args.task))



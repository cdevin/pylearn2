import os
import argparse, fnmatch
import numpy as np
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def svhn(submit = False, make = False):
    """
    SVHN
    """
    state = DD()
    with open('exp/svhn.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN'
    #state.orig_path = '/RQexec/mirzameh/data/SVHN/'
    state.orig_path = '/scratch/mmirza/data/SVHN/channel/'
    state.data_path = '/tmp/mmirza/SVHN/'
    num_exp = 25
    if submit:
        TABLE_NAME = "remu_svhn_monk"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/remu_svhn/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 2, 7])

    for i in xrange(num_exp):
        state.y_max_col_norm = rng.uniform(1., 3.)

        channel_options = [32, 64, 80, 96, 128, 256]
        state.h0_num_channels = channel_options[rng.randint(len(channel_options))]
        state.h1_num_channels = channel_options[rng.randint(len(channel_options))]
        state.h2_num_channels = channel_options[rng.randint(len(channel_options))]

        state.h0_num_pieces = rng.randint(2, 5)
        state.h1_num_pieces = rng.randint(2, 5)
        state.h2_num_pieces = rng.randint(2, 5)

        state.h0_kernel_shape = rng.randint(2, 8)
        state.h1_kernel_shape = rng.randint(2, 7)
        state.h2_kernel_shape = rng.randint(2, 6)

        state.h0_pool_shape = rng.randint(2, 4)
        state.h1_pool_shape = rng.randint(2, 4)
        state.h2_pool_shape = rng.randint(2, 4)

        state.h0_pool_stride = rng.randint(1, state.h0_pool_shape)
        state.h1_pool_stride = rng.randint(1, state.h1_pool_shape)
        state.h2_pool_stride = rng.randint(1, state.h2_pool_shape)

        state.h3_num_units = rng.randint(200, 1000)
        state.h3_num_pieces = rng.randint(2, 5)
        state.h3_w_scale= 1.
        state.h3_slope_scale = 1.
        state.h4_num_units = rng.randint(100, 1000)
        state.h4_num_pieces = rng.randint(2, 5)
        state.h4_w_scale= 1.
        state.h4_slope_scale = 1.


        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.momentum_saturate = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .9)
        state.lr_sat =rng.randint(50, 200)
        state.lr_decay = 10. ** rng.uniform(-3, -1)

        state.h0_islope = 10 ** rng.uniform(0, -2.)
        state.h1_islope = 10 ** rng.uniform(0, -2.)
        state.h2_islope = 10 ** rng.uniform(0, -2.)
        state.h3_islope = 10 ** rng.uniform(0, -2.)
        state.h4_islope = 10 ** rng.uniform(0, -2.)

        def random_init_string(low = -2.3, high = -1.):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string()
        state.h1_init = random_init_string()
        state.h2_init = random_init_string()
        state.h3_init = random_init_string()
        state.h4_init = random_init_string()
        if rng.randint(2):
            state.y_init = "sparse_init: {}".format([0, 10, 100][rng.randint(3)])
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

def svhn_c(submit = False, make = False):
    """
    SVHN
    """
    state = DD()
    with open('exp/svhn_continue.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'SVHN'
    #state.orig_path = '/RQexec/mirzameh/data/SVHN/'
    state.orig_path = '/scratch/mmirza/data/SVHN/channel/'
    state.data_path = '/tmp/mmirza/SVHN/'
    num_exp = 20
    if submit:
        TABLE_NAME = "remu_svhn_c"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/remu_svhn/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 2, 7])

    for i in xrange(num_exp):

        state.learning_rate = 10. ** rng.uniform(0., -4)
        state.momentum_saturate = rng.randint(1, 100)
        state.init_momentum = rng.uniform(.6, .8)
        state.final_momentum = rng.uniform(state.init_momentum, .8)
        state.lr_sat =rng.randint(1, 100)
        state.lr_decay = 10. ** rng.uniform(-3, 0)

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
    parser.add_argument('-t', '--task', choices = ['svhn', 'svhn_c'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    parser.add_argument('-m', '--make', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'svhn':
        svhn(args.submit, args.make)
    elif args.task == 'svhn_c':
        svhn_c(args.submit, args.make)
    else:
        raise ValueError or("Wrong task optipns {}".fromat(args.task))



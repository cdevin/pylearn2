import os
import argparse, fnmatch
import numpy as np
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def composite_relu(submit = False, make = False):
    """
    2 sparse linear composite layer
    """
    state = DD()
    with open('exp/penntree_maxout_local_relu_composite.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 7
    state.batch_size = 640
    state.embed_use_bias = 0
    state.h1_use_bias = 0
    state.converter_num_channel = 3
    num_exp = 20
    if submit:
        TABLE_NAME = "pentree_sparse_local_relu_composite"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_sparse_relu_composite_7/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_max_col_norm = rng.uniform(1., 4.)
        #state.h0_max_col_norm = 1.
        state.h0_min_col_norm = 0
        state.h1_col_norm = rng.uniform(1., 4.)
        state.h2_col_norm = rng.uniform(1., 2.5)
        state.h3_col_norm = rng.uniform(1., 2.5)
        state.h4_col_norm = rng.uniform(1., 2.5)
        state.y_col_norm = rng.uniform(1., 3.)

        channel_options = [16]
        state.h2_channels = channel_options[rng.randint(len(channel_options))]

        state.embed_dim = rng.randint(10, 300)
        state.img_shape = rng.randint(15, 50)
        state.linear_dim = (state.img_shape ** 2) * state.converter_num_channel
        state.h2_num_pieces = rng.randint(2, 3)
        state.h2_kernel_shape = rng.randint(2, 3)
        state.h3_num_pieces = rng.randint(2, 3)
        state.h3_units = rng.randint(50, 200)
        state.h3_pieces = rng.randint(2, 3)
        state.h4_units = rng.randint(100, 1000)
        state.h4_pieces = rng.randint(2, 3)
        state.learning_rate = 10. ** rng.uniform(1., -1)
        state.m_sat = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .7)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

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

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['sparse',
            'channel', 'linear', 'linear2', 'dense', 'maxout',
            'wiki_sparse2', 'composite', 'relu', 'composite_relu'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    parser.add_argument('-m', '--make', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'sparse':
        sparse(args.submit)
    else:
        raise ValueError or("Wrong task optipns {}".fromat(args.task))



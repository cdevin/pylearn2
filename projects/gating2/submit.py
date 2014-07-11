import os
import argparse, fnmatch
import numpy as np
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()


def unigram(submit = False, make = False, num_exp = 20):
    state = DD()
    with open('exp/nce_lbl_unigram.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    if submit:
        TABLE_NAME = "nce_penn"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/nce/")
    PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.max_col_norm = rng.uniform(1., 3.)

        state.batch_size = rng.randint(50, 500)
        state.dim = rng.randint(50, 1000)
        state.context_len = rng.randint(2,15)
        state.nce_k = rng.randint(2, 30)

        state.learning_rate = 10. ** rng.uniform(1., -3)
        state.momentum_saturate = rng.randint(20, 250)
        state.final_momentum = rng.uniform(.5, .9)
        state.lr_decay = 10. ** rng.uniform(-3, -1)
        state.lr_saturate = rng.randint(20, 250)

        def random_init_string():
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.irange = random_init_string()
        #if rng.randint(2):
            #state.y_init = "sparse_init: 0"
        #else:
            #state.y_init = random_init_string()

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
    parser.add_argument('-t', '--task', choices = ['unigram'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    parser.add_argument('-m', '--make', default = False, action='store_true')
    parser.add_argument('-n', '--num_exp', default = 20, type = int)
    args = parser.parse_args()

    if args.task == 'unigram':
        print args.submit
        unigram(args.submit, args.make, args.num_exp)
    else:
        raise ValueError or("Wrong task optipns {}".fromat(args.task))



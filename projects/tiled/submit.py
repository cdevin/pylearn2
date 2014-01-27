import os
import argparse, fnmatch
import numpy as np
from jobman import DD, flatten, api0, sql
from noisylearn.projects.train import experiment
from my_utils.config import get_data_path, get_result_path
from pylearn2.utils.string_utils import preprocess

DATA_PATH = get_data_path()

def sparse(submit = False):
    state = DD()
    with open('exp/penntree_maxout_local_2.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 6
    state.embed_dim = 96
    state.img_shape = 24
    state.l1_num_pieces = 2
    state.kernel_shape = 2
    state.l2_units = 50
    state.l2_pieces = 2
    state.batch_size = 128
    state.learning_rate = 0.1
    state.m_stat = 100
    state.final_momentum = 0.7
    state.lr_sat = 100
    state.decay = 0.1
    num_exp = 10
    if submit:
        TABLE_NAME = "pentree_sparse"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_sparse/")

    rng = np.random.RandomState([2014, 1, 10])


    for i in xrange(num_exp):
        state.h0_col_norm = rng.uniform(1., 5.)
        state.h1_col_norm = rng.uniform(1., 5.)
        state.h2_col_norm = rng.uniform(1., 5.)
        state.y_col_norm = rng.uniform(1., 5.)

        state.l1_num_pieces = rng.randint(2, 5)
        state.kernel_shape = rng.randint(2, 6)
        state.l2_units = rng.randint(50, 300)
        state.l2_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(0., -.4)
        state.m_sat = rng.randint(2, 500)
        state.final_momentum = rng.uniform(.5, .9)
        state.lr_sat =rng.randint(150, 500)
        state.decay = 10. ** rng.uniform(-3, -1)

        def random_init_string():
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string()
        state.h1_init = random_init_string()
        state.h2_init = random_init_string()
        if rng.randint(2):
            state.y_init = "sparse_init: 0"
        else:
            state.y_init = random_init_string()

        if submit:
            sql.insert_job(experiment, flatten(state), db)
        else:
            experiment(state, None)

    db.createView(TABLE_NAME + '_view')

def sparse_maxout(submit = False, make = False):
    state = DD()
    with open('exp/penntree_maxout_local_maxout.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 6
    state.embed_dim = 96
    state.img_shape = 24
    state.l1_num_pieces = 2
    state.kernel_shape = 2
    state.l2_units = 50
    state.l2_pieces = 2
    state.batch_size = 128
    state.learning_rate = 0.1
    state.m_stat = 100
    state.final_momentum = 0.7
    state.lr_sat = 100
    state.decay = 0.1
    num_exp = 30
    if submit:
        TABLE_NAME = "pentree_sparse_local_maxout"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/penntree_maxout_local_maxout/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_col_norm = rng.uniform(1., 2.)
        state.h1_col_norm = rng.uniform(1., 2.)
        state.h2_col_norm = rng.uniform(1., 2.)
        state.h3_col_norm = rng.uniform(2., 4.)
        state.y_col_norm = rng.uniform(3., 7.)

        state.embed_dim = rng.randint(50, 150)
        state.img_shape = rng.randint(10, 30)
        state.l1_units = state.img_shape ** 2
        state.l1_pieces = rng.randint(2, 5)
        state.l2_num_pieces = rng.randint(2, 5)
        state.kernel_shape = rng.randint(2, 6)
        state.l3_units = rng.randint(50, 200)
        state.l3_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.m_sat = rng.randint(2, 200)
        state.final_momentum = rng.uniform(.5, .7)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

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

def sparse_linear(submit = False, make = False):
    """
    2 sparse linear layer
    """
    state = DD()
    with open('exp/penntree_maxout_local_linear.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 5
    state.batch_size = 640
    state.embed_use_bias = 0
    state.h1_use_bias = 0
    state.converter_num_channel = 3
    num_exp = 20
    if submit:
        TABLE_NAME = "pentree_sparse_local_linear2_new_gui"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_sparse_local1/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_max_col_norm = rng.uniform(1., 2.)
        #state.h0_max_col_norm = 1.
        state.h0_min_col_norm = 0
        state.h1_col_norm = rng.uniform(1., 2.)
        state.h2_col_norm = rng.uniform(2., 3.5)
        state.h3_col_norm = rng.uniform(2., 3.5)
        state.h4_col_norm = rng.uniform(1., 2.5)
        state.y_col_norm = rng.uniform(3., 8.)

        channel_options = [16]
        state.h2_channels = channel_options[rng.randint(len(channel_options))]
        state.h3_channels = channel_options[rng.randint(len(channel_options))]

        state.embed_dim = rng.randint(200, 1000)
        state.img_shape = rng.randint(10, 40)
        state.linear_dim = (state.img_shape ** 2) * state.converter_num_channel
        state.h2_num_pieces = rng.randint(2, 5)
        state.h2_kernel_shape = rng.randint(2, 5)
        state.h3_num_pieces = rng.randint(2, 5)
        state.h3_kernel_shape = rng.randint(2, 4)
        state.h4_units = rng.randint(100, 1000)
        state.h4_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.m_sat = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .75)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

        def random_init_string(low = -2.3, high = -1.):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string(-4, -2)
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

def sparse_linear2(submit = False, make = False):
    """
    2 sparse linear layer
    """
    state = DD()
    with open('exp/penntree_maxout_local_linear2.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 5
    state.batch_size = 640
    state.embed_use_bias = 0
    state.h1_use_bias = 0
    state.converter_num_channel = 3
    num_exp = 20
    if submit:
        TABLE_NAME = "pentree_sparse_local_linear2_new_gui"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_sparse_local2_5/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_max_col_norm = rng.uniform(1., 2.)
        #state.h0_max_col_norm = 1.
        state.h0_min_col_norm = 0
        state.h1_col_norm = rng.uniform(1., 2.)
        state.h2_col_norm = rng.uniform(2., 3.5)
        state.h3_col_norm = rng.uniform(2., 3.5)
        state.h4_col_norm = rng.uniform(1., 2.5)
        state.y_col_norm = rng.uniform(3., 8.)

        channel_options = [16]
        state.h2_channels = channel_options[rng.randint(len(channel_options))]
        state.h3_channels = channel_options[rng.randint(len(channel_options))]

        state.embed_dim = rng.randint(10, 300)
        state.img_shape = rng.randint(10, 40)
        state.linear_dim = (state.img_shape ** 2) * state.converter_num_channel
        state.h2_num_pieces = rng.randint(2, 5)
        state.h2_kernel_shape = rng.randint(2, 5)
        state.h3_num_pieces = rng.randint(2, 5)
        state.h3_kernel_shape = rng.randint(2, 4)
        state.h4_units = rng.randint(700, 1200)
        state.h4_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.m_sat = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .75)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

        def random_init_string(low = -2.3, high = -1.):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string(-4, -2)
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

def sparse_linear2_wiki(submit = False, make = False):
    """
    2 sparse linear layer
    """
    state = DD()
    with open('exp/wikipedia_maxout_local_linear2.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'wikipedia'
    state.seq_len = 100
    state.batch_size = 1920
    state.embed_use_bias = 0
    state.h1_use_bias = 0
    state.monitoring_batches = np.ceil(4999900 / state.batch_size)
    num_exp = 20
    if submit:
        TABLE_NAME = "wikipeida_sparse_local_linear2_monk"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/wikipedia_sparse_local2/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_col_norm = rng.uniform(1., 4.)
        state.h1_col_norm = rng.uniform(1., 4.)
        state.h2_col_norm = rng.uniform(2., 3.5)
        state.h3_col_norm = rng.uniform(2., 3.5)
        state.h4_col_norm = rng.uniform(1., 2.5)
        state.y_col_norm = rng.uniform(1., 8.)

        channel_options = [16]
        state.h2_channels = channel_options[rng.randint(len(channel_options))]
        state.h3_channels = channel_options[rng.randint(len(channel_options))]

        state.embed_dim = rng.randint(5, 15)
        state.img_shape = rng.randint(20, 40)
        state.linear_dim = state.img_shape ** 2
        state.h2_num_pieces = rng.randint(3, 6)
        state.h2_kernel_shape = rng.randint(4, 6)
        state.h3_num_pieces = rng.randint(2, 5)
        state.h3_kernel_shape = rng.randint(4, 6)
        state.h4_units = rng.randint(100, 1000)
        state.h4_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.m_sat = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .8)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

        #state.h2_pad = state.h2_kernel_shape - 2
        #state.h3_pad = state.h3_kernel_shape - 2
        state.h2_pad = 0
        state.h3_pad = 0

        def random_init_string(low = -2.3, high = -1.):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string(-4, -2)
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

def channel(submit = False, make = False):
    state = DD()
    with open('exp/penntree_maxout_local_multichannel.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 6
    state.img_shape = 24
    state.l1_num_pieces = 2
    state.kernel_shape = 2
    state.l2_units = 50
    state.l2_pieces = 2
    state.batch_size = 128
    state.learning_rate = 0.1
    state.m_stat = 100
    state.final_momentum = 0.7
    state.lr_sat = 100
    state.decay = 0.1
    num_exp = 25
    if submit:
        TABLE_NAME = "pentree_sparse_bri"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_channel/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])


    for i in xrange(num_exp):
        state.img_shape = rng.randint(10, 50)
        state.h0_col_norm = rng.uniform(1., 3.)
        state.h1_col_norm = rng.uniform(1., 2.)
        state.h2_col_norm = rng.uniform(2., 5.)
        state.y_col_norm = rng.uniform(3., 10.)

        state.l1_num_pieces = rng.randint(2, 5)
        state.kernel_shape = rng.randint(2, 4)
        #state.kernel_stride = min(state.kernel_shape - 1, rng.randint(2, 6))
        state.kernel_stride = 1
        state.l2_units = rng.randint(50, 200)
        state.l2_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(0., -.2)
        state.m_sat = rng.randint(2, 500)
        state.final_momentum = rng.uniform(.5, .7)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

        def random_init_string():
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string()
        state.h1_init = random_init_string()
        state.h2_init = random_init_string()
        if rng.randint(2):
            state.y_init = "sparse_init: 0"
        else:
            state.y_init = random_init_string()

        if make:
            state.save_path = os.path.join(PATH, str(i))
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

def dense(submit = False, make = False):
    "densly connected one layer"
    state = DD()
    with open('exp/penntree_dense.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.batch_size = 640
    num_exp = 30
    if submit:
        TABLE_NAME = "pentree_dense_2"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_dense/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_col_norm = rng.uniform(1., 4.)
	state.h0_min_col_norm =0
	state.embed_use_bias = 0
        state.h1_col_norm = rng.uniform(1., 3.)
        state.h2_col_norm = rng.uniform(1., 3.)
        state.y_col_norm = rng.uniform(3., 9.)

        state.seq_len = rng.randint(5,7)
        state.embed_dim = rng.randint(10,1000)
        state.h1_num_pieces = rng.randint(2, 5)
        state.h1_num_units = rng.randint(200, 1500)
        state.h2_num_pieces = rng.randint(2, 5)
        state.h2_num_units = rng.randint(200, 1500)
        state.learning_rate = 10. ** rng.uniform(1., -3)
        state.m_sat = rng.randint(50, 150)
        state.final_momentum = rng.uniform(.5, .7)
        state.lr_sat =rng.randint(50, 150)
        state.decay = 10. ** rng.uniform(-3, -1)

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

def composite(submit = False, make = False):
    """
    2 sparse linear composite layer
    """
    state = DD()
    with open('exp/penntree_maxout_local_linear_composite.yaml') as ymtmp:
        state.yaml_string = ymtmp.read()

    state.db = 'penntree'
    state.seq_len = 5
    state.batch_size = 640
    state.embed_use_bias = 0
    state.h1_use_bias = 0
    state.converter_num_channel = 1
    num_exp = 20
    if submit:
        TABLE_NAME = "pentree_sparse_local_linear_composite"
        db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
        state.save_path = './'
    else:
        state.save_path = preprocess("${PYLEARN2_EXP_RESULTS}/pentree_sparse_local_composite/")
        PATH = state.save_path

    rng = np.random.RandomState([2014, 1, 15])

    for i in xrange(num_exp):
        state.h0_max_col_norm = rng.uniform(1., 2.)
        #state.h0_max_col_norm = 1.
        state.h0_min_col_norm = 0
        state.h1_col_norm = rng.uniform(1., 2.)
        state.h2_col_norm = rng.uniform(2., 3.5)
        state.h3_col_norm = rng.uniform(2., 3.5)
        state.h4_col_norm = rng.uniform(1., 2.5)
        state.y_col_norm = rng.uniform(3., 8.)

        channel_options = [16]
        state.h2_channels = channel_options[rng.randint(len(channel_options))]

        state.embed_dim = rng.randint(10, 300)
        state.img_shape = rng.randint(10, 40)
        state.linear_dim = (state.img_shape ** 2) * state.converter_num_channel
        state.h2_num_pieces = rng.randint(2, 5)
        state.h2_kernel_shape = rng.randint(2, 5)
        state.h3_num_pieces = rng.randint(2, 5)
        state.h3_units = rng.randint(50, 100)
        state.h3_pieces = rng.randint(2, 5)
        state.h4_units = rng.randint(100, 1000)
        state.h4_pieces = rng.randint(2, 5)
        state.learning_rate = 10. ** rng.uniform(1., -2)
        state.m_sat = rng.randint(50, 200)
        state.final_momentum = rng.uniform(.6, .75)
        state.lr_sat =rng.randint(50, 200)
        state.decay = 10. ** rng.uniform(-3, -1)

        def random_init_string(low = -2.3, high = -1.):
            irange = 10. ** rng.uniform(-2.3, -1.)
            return "irange: " + str(irange)

        state.h0_init = random_init_string(-4, -2)
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
            'wiki_sparse2', 'composite'])
    parser.add_argument('-s', '--submit', default = False, action='store_true')
    parser.add_argument('-m', '--make', default = False, action='store_true')
    args = parser.parse_args()

    if args.task == 'sparse':
        sparse(args.submit)
    elif args.task == 'linear':
        sparse_linear(args.submit, args.make)
    elif args.task == 'maxout':
        sparse_maxout(args.submit, args.make)
    elif args.task == 'linear2':
        sparse_linear2(args.submit, args.make)
    elif args.task == 'dense':
        dense(args.submit, args.make)
    elif args.task == 'channel':
        channel(args.submit, args.make)
    elif args.task == 'wiki_sparse2':
        sparse_linear2_wiki(args.submit, args.make)
    elif args.task == 'composite':
        composite(args.submit, args.make)
    else:
        raise ValueError or("Wrong task optipns {}".fromat(args.task))



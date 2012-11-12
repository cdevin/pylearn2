import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from train import DATA_PATH, RESULT_PATH, train_1layer_yaml_string
from train import experiment as train_experiment
from classify import experiment as classify_experiment
from l2_svm import experiment as l2_svm_experiment
from train_supervised import experiment as mlp_experiment
from utils.config import get_experiment_path

def train_layer1_mnist():

    state = DD()

    state.data_path = os.path.join(DATA_PATH, "mnist/pylearn2/train.pkl")
    state.nvis = 784
    state.nhid = 1000
    state.learning_rate = 0.0001
    state.grow_amt = 1.001
    state.shrink_amt = 0.009
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.start_momentum = 10
    state.saturate_momentum = 30
    state.w_l1_ratio = 0.001
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.input_corruption_level = 0.2
    state.hidden_corruption_level = 0.5
    state.batch_size = 20
    state.monitoring_batches = 3000
    state.normal_penalty = 1
    state.max_epochs = 300
    state.save_name = "mnist_l1_"
    state.save_freq = 10
    state.yaml_string = train_1layer_yaml_string


    ind = 0
    TABLE_NAME = "smooth_dropout_mnist_l1"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.01, 0.001, 0.0001]:
        for wl1 in [0, 0.0001]:
            for in_corr in [0.2, 0.5]:
                for hid_corr in [0.2, 0.5, 0.7]:
                        state.learning_rate = lr
                        state.grow_amt = 1. + 10. ** numpy.log10(lr)
                        state.shrink_amt = 1. - 10. ** numpy.log10(lr)
                        state.w_l1_ratio = wl1
                        state.input_corruption_level = in_corr
                        state.hidden_corruption_level = hid_corr
                        sql.insert_job(train_experiment, flatten(state), db)
                        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def train_layer1_cifar():

    state = DD()

    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/train.pkl")
    state.nvis = 32 * 32 * 3
    state.nhid = 1000
    state.learning_rate = 0.0001
    state.grow_amt = 1.001
    state.shrink_amt = 0.009
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.start_momentum = 10
    state.saturate_momentum = 30
    state.w_l1_ratio = 0.001
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.input_corruption_level = 0.2
    state.hidden_corruption_level = 0.5
    state.batch_size = 20
    state.monitoring_batches = 2500
    state.normal_penalty = 1
    state.max_epochs = 300
    state.save_name = "cifar_l1_"
    state.save_freq = 20
    state.yaml_string = train_1layer_yaml_string


    ind = 0
    TABLE_NAME = "smooth_dropout_cifar_l1"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.01, 0.001, 0.0001]:
        for wl1 in [0, 0.001]:
            for in_corr in [0.0,  0.2, 0.5]:
                for hid_corr in [0.0, 0.2, 0.5, 0.7]:
                        state.learning_rate = lr
                        state.grow_amt = 1. + 10. ** numpy.log10(lr)
                        state.shrink_amt = 1. - 10. ** numpy.log10(lr)
                        state.w_l1_ratio = wl1
                        state.input_corruption_level = in_corr
                        state.hidden_corruption_level = hid_corr
                        sql.insert_job(train_experiment, flatten(state), db)
                        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def classify_mnist():

    state = DD()

    state.model_path = os.path.join(get_experiment_path(), "smooth_dropout_mnist_l1")
    state.n_folds = 1
    state.scale = True
    state.norm = True
    state.dataset = 'mnist'
    state.method = 'svm'
    state.lr_vals = [1000000]

    matches = []
    for root, dirnames, filenames in os.walk(state.model_path):
        for filename in fnmatch.filter(filenames, '*_299.pkl'):
            matches.append(os.path.join(root, filename))

    ind = 0
    TABLE_NAME = "smooth_dropout_mnist_l1_svm"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for item in matches:
        state.model_f = item
        sql.insert_job(classify_experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def classify_mnist_l2_svm():

    state = DD()

    search_path = os.path.join(get_experiment_path(), "smooth_dropout_mnist_l1")
    state.labels_path = os.path.join(get_experiment_path(), "smooth_dropout_mnist_l1/labels.mat")
    state.standardize = 'False'
    state.dataset = 'mnist'
    state.method = 'svm'
    state.c_vals = '[1000 10000 100000 1000000 10000000]'

    matches = []
    for root, dirnames, filenames in os.walk(search_path):
        for filename in fnmatch.filter(filenames, '*.mat'):
            matches.append(os.path.join(root, filename))

    ind = 0
    TABLE_NAME = "smooth_dropout_mnist_l1_svm_l2"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for item in matches:
        state.data_path = item
        sql.insert_job(l2_svm_experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def classify_cifar_l1_svm():

    state = DD()

    search_path = os.path.join(get_experiment_path(), "smooth_dropout_cifar_l1")
    state.standardize = 'False'
    state.dataset = 'cifar'
    state.c_vals = '[10000 100000 1000000 10000000]'

    matches = []
    for root, dirnames, filenames in os.walk(search_path):
        for filename in fnmatch.filter(filenames, '*.pkl'):
            matches.append(os.path.join(root, filename))

    ind = 0
    TABLE_NAME = "smooth_dropout_cifar_l1_svm"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for item in matches:
        state.data_path = item
        sql.insert_job(l2_svm_experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def mlp_cifar():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/")
    state.shuffle = False
    state.dataset = 'cifar10'
    state.nouts = 10
    state.act_enc = "rectifier"
    state.scale = False
    state.norm = False
    state.model = 'mlp'
    state.nepochs = 1000
    state.lr = 0.05
    state.lr_shrink_time = 700
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 100
    state.batch_size = 100
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.05
    state.n_units = [32*32*3, 1024, 1024, 1024]
    state.group_sizes = [128, 128]
    state.gaussian_corruption_levels = [0.5, 0.5, 0.5]
    state.binomial_corruption_levels = [0.5, 0.5, 0.5, 0.5]
    state.group_corruption_levels = None
    state.save_frequency = 50
    state.save_name = "cifar10_l2.pkl"
    state.fold = 0
    state.description = "Data rescale=True, center = True, global=standardize\
            weight init, normal, bias +1"

    ind = 0
    TABLE_NAME = "sd_mlp_cifar_2l_s_3"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.01, 0.001]:
        for g0 in [0.1, 0.3, 0.5, 0.7, 0.9, 0.0]:
            for g1 in [0.0]:
                for g2 in [0.0]:
                    for g3 in [0.0]:
                        state.lr = lr
                        #state.binomial_corruption_levels = [l0_corr, l1_corr, l2_corr]
                        state.gaussian_corruption_levels = [g0, g1, g2, g3]
                        sql.insert_job(mlp_experiment, flatten(state), db)
                        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def mlp_cifar_g():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/")
    state.shuffle = False
    state.dataset = 'cifar10'
    state.act_enc = "rectifier"
    state.scale = True
    state.norm = False
    state.nepochs = 1000
    state.lr = 0.05
    state.lr_shrink_time = 60
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 100
    state.batch_size = 50
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.1
    state.n_units = [32*32*3, 1024, 1024, 1024]
    state.group_sizes = [128, 128, 128]
    state.gaussian_corruption_levels = [0.5, 0.5, 0.5, 0.5]
    state.binomial_corruption_levels = [0.0, 0.3, 0.3]
    state.group_corruption_levels = [0.0, 0.2]
    state.save_frequency = 50
    state.save_name = "cifar_l5.pkl"

    ind = 0
    TABLE_NAME = "sd_mlp_cifar_2l_g"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.01, 0.001]:
        for g_0 in [0.0, 0.3]:
            for g_1 in [0.0, 0.3]:
                for g_2 in [0.0, 0.3]:
                    for act_l1 in [0.0, 0.1]:
                        state.lr = lr
                        #state.binomial_corruption_levels = [i_0, i_1]
                        state.group_corruption_levels = [g_0, g_1, g_2]
                        state.act_l1_ratio = act_l1
                        sql.insert_job(mlp_experiment, flatten(state), db)
                        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def mlp_cifar100():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "cifar100/pylearn2/")
    state.shuffle = False
    state.dataset = 'cifar100'
    state.nouts = 100
    state.act_enc = "rectifier"
    state.scale = False
    state.norm = False
    state.model = 'mlp'
    state.nepochs = 1000
    state.lr = 0.05
    state.lr_shrink_time = 60
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 100
    state.batch_size = 200
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.1
    state.n_units = [32*32*3, 1024, 1024, 1024]
    state.group_sizes = [128, 128, 128, 128]
    state.gaussian_corruption_levels = [0.5, 0.5, 0.5, 0.5]
    state.binomial_corruption_levels = [0.0, 0.3, 0.3]
    state.group_corruption_levels = None
    state.save_frequency = 50
    state.save_name = "cifar100_l3.pkl"
    state.fold = 0

    ind = 0
    TABLE_NAME = "sd_mlp_cifar100_3l_2"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1, 0.01]:
        for gauss in [[0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]]:
            for l0 in [0.0, 0.5]:
                for l1 in [0.0, 0.5]:
                    for l2 in [0.0, 0.5]:
                        for l3 in [0.0, 0.5]:
                            state.lr = lr
                            state.gaussian_corruption_levels = gauss
                            state.binomial_corruption_levels = [l0, l1, l2, l3]
                            sql.insert_job(mlp_experiment, flatten(state), db)
                            ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jo bs submitted".format(ind)

def mlp_mnist():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "mnist/pylearn2/")
    state.shuffle = False
    state.dataset = 'mnist'
    state.nouts = 10
    state.act_enc = "rectifier"
    state.scale = True
    state.norm = False
    state.model = 'mlp'
    state.nepochs = 1000
    state.lr = 0.05
    state.lr_shrink_time = 60
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 50
    state.momentum_inc_end = 100
    state.batch_size = 200
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.1
    state.n_units = [28*28, 1024, 1024]
    state.group_sizes = [128, 128, 128, 128]
    state.gaussian_corruption_levels = [0.5, 0.5, 0.5, 0.5]
    state.binomial_corruption_levels = [0.5, 0.5, 0.0]
    state.group_corruption_levels = None
    state.save_frequency = 50
    state.save_name = "mnist_l3.pkl"
    state.fold = 0

    ind = 0
    TABLE_NAME = "sd_mlp_mnist_2l_2"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.01]:
        for g0 in [0.1, 0.3, 0.5, 0.7, 0.9]:
            for g1 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                for g2 in [0.1, 0.3, 0.5, 0.7, 0.9]:
                    state.lr = lr
                    #state.binomial_corruption_levels = [l0_corr, l1_corr, l2_corr]
                    state.gaussian_corruption_levels = [g0, g1, g2]
                    sql.insert_job(mlp_experiment, flatten(state), db)
                    ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)

def mlp_timit():

    state = DD()
    state.data_path = os.path.join(DATA_PATH, "timit/pylearn2/")
    state.shuffle = False
    state.dataset = 'timit'
    state.act_enc = "rectifier"
    state.scale = False
    state.norm = False
    state.nepochs = 1000
    state.lr = 0.05
    state.lr_shrink_time = 50
    state.lr_dc_rate = 0.01
    state.enable_momentum = True
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.momentum_inc_start = 20
    state.momentum_inc_end = 50
    state.batch_size = 50
    state.w_l1_ratio = 0.0
    state.act_l1_ratio = 0.0
    state.irange = 0.01
    state.n_units = [384, 1024, 1024]
    state.group_sizes = [128, 128]
    state.gaussian_corruption_levels = [0.5, 0.5, 0.5]
    state.binomial_corruption_levels = [0.0, 0.5]
    state.group_corruption_levels = None
    state.save_frequency = 50
    state.save_name = "timit_l2.pkl"

    ind = 0
    TABLE_NAME = "sd_mlp_timit_2l"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.1, 0.01]:
        for g_0 in [0.0, 0.5]:
            for g_1 in [0.0, 0.5]:
                for g_2 in [0.0, 0.5]:
                    for l1_corr in [0.0, 0.5]:
                        for l2_corr in [0.0, 0.5]:
                            for act_l1 in [0.0, 0.1]:
                                state.lr = lr
                                state.binomial_corruption_levels = [l1_corr, l2_corr]
                                state.gaussian_corruption_levels = [g_0, g_1, g_2]
                                state.act_l1_ratio = act_l1
                                sql.insert_job(mlp_experiment, flatten(state), db)
                                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Albedo trainer submitter')
    parser.add_argument('-t', '--task', choices = ['layer1_mnist', 'classify_mnist',
                    'layer1_cifar', 'classify_cifar', 'classify_mnist_l2_svm', 'mlp_timit',
                    'classify_cifar_l1_svm', 'mlp_cifar', 'mlp_cifar_g', 'mlp_cifar100', 'mlp_mnist'])
    args = parser.parse_args()

    if args.task == 'layer1_mnist':
        train_layer1_mnist()
    elif args.task == 'classify_mnist':
        classify_mnist()
    elif args.task == 'classify_mnist_l2_svm':
        classify_mnist_l2_svm()
    elif args.task == 'classify_cifar_l1_svm':
        classify_cifar_l1_svm()
    elif args.task == 'layer1_cifar':
        train_layer1_cifar()
    elif args.task == 'classify_cifar':
        classify_cifar()
    elif args.task == 'mlp_cifar':
        mlp_cifar()
    elif args.task == 'mlp_cifar_g':
        mlp_cifar_g()
    elif args.task == 'mlp_cifar100':
        mlp_cifar100()
    elif args.task == 'mlp_mnist':
        mlp_mnist()
    elif args.task == 'mlp_timit':
        mlp_timit()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



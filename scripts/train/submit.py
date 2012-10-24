import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from train import DATA_PATH, RESULT_PATH, train_1layer_yaml_string
from train import experiment as train_experiment
from lg import experiment as mlp_experiment
from utils.config import get_experiment_path

def train_layer1():

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

def classify():

    state = DD()

    state.model_path = os.path.join(get_experiment_path(), "smooth_dropout_mnist_l1")
    state.n_folds = 1
    state.scale = True
    state.lr_vals = [0.01, 0.001]


    matches = []
    for root, dirnames, filenames in os.walk(state.model_path):
        for filename in fnmatch.filter(filenames, '*_299.pkl'):
            matches.append(os.path.join(root, filename))

    ind = 0
    TABLE_NAME = "smooth_dropout_mnist_l1_mlp"
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for item in matches:
        state.model_f = item
        sql.insert_job(mlp_experiment, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'Albedo trainer submitter')
    parser.add_argument('-t', '--task', choices = ['layer1', 'classify'])
    args = parser.parse_args()

    if args.task == 'layer1':
        train_layer1()
    elif args.task == 'classify':
        classify()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



import os
import argparse, fnmatch
import numpy
from jobman import DD, flatten, api0, sql
from noisy_encoder.scripts.icml.train import train_yaml
from noisy_encoder.scripts.icml.train import experiment
from my_utils.config import get_data_path, get_result_path

DATA_PATH = get_data_path()

def svhn_ian():
    state = DD()
    state.yaml_string = train_yaml

    state.data_path = '/tmp/data/SVHN/'
    state.num_channels_1 = 32
    state.num_channels_2 = 32
    state.num_channels_3 = 64
    state.max_kernel_norm_1 = 1.5
    state.max_kernel_norm_2 = 2.9
    state.max_kernel_norm_3 = 2.9
    state.learning_rate = 0.05
    state.decay_factor = 1.000004
    state.momentum_saturate = 20
    state.final_momentum = 0.9
    state.max_epochs = 300
    state.best_save_path = "best.pkl"
    state.save_path = "current.pkl"

    ind = 0
    TABLE_NAME = "ian_svhn"
    db = api0.open_db("postgres://mirzamom:pishy83@opter.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)
    for lr in [0.05, 0.005]:
        for ch1, ch2, ch3 in zip([64, 128, 128], [64, 128, 256], [128, 256, 512]):
            for decay in [1.000004, 1.00004]:
                if lr == 0.005:
                    decay = 1.000004
                state.learning_rate = lr
                state.num_channels_1 = ch1
                state.num_channels_2 = ch2
                state.num_channels_3 = ch3
                state.decay_factor = decay
                sql.insert_job(experiment, flatten(state), db)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print "{} jobs submitted".format(ind)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'job submitter')
    parser.add_argument('-t', '--task', choices = ['svhn_ian'])
    args = parser.parse_args()

    if args.task == 'svhn_ian':
        svhn_ian()
    else:
        raise ValueError("Wrong task optipns {}".fromat(args.task))



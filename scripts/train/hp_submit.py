import os
import argparse
from train_supervised_hp import experiment
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, trials_from_docs
from hyperopt.mongoexp import MongoTrials
from utils.config import get_data_path, get_result_path

RESULT_PATH = get_result_path()
DATA_PATH = get_data_path()


def tfd_conv(max_evals = 1):
    space = {}
    space['lr_init'] = hp.loguniform('lr_init',0e-4, 0.1)
    space['shape_1'] = [48, 48]
    space['kernel_shape_1'] = hp.choice('kernel_choice', [7, 7 ])

    space["dataset"] = 'tfd'
    space["fold"] = 0
    space["data_path"] = os.path.join(DATA_PATH, "faces/TFD/pylearn2/{}/".format(space["fold"]))
    #space["data_path = os.path.join(DATA_PATH, "faces/tfd_lisa/pylearn2/")
    space["scale"] = False
    space["norm"] = False
    space["shuffle"] = False
    space["train_alg"] = "sgd"
    space["nepochs"] = 3
    space["lr_params"] = {'shrink_time': 10, 'init_value' : space['lr_init'], 'dc_rate' : 0.001}
    space["enable_momentum"] = True
    space["momentum_params"] = {'inc_start' : 30, 'inc_end' : 70, 'init_value' : 0.5, 'final_value' : 0.9}
    space["batch_size"] = 20
    space["w_l1_ratio"] = 0.000
    space["act_l1_ratio"] = 0.0
    space["save_frequency"] = 100
    space["save_name"] = os.path.join(RESULT_PATH, "naenc/tfd/tfd_gpu.pkl")
    space["coeffs"] = {'w_l1' : 0.0, 'w_l2' : 0e-06}

    # model params
    space["model"] = 'conv'
    space["conv_layers"] = [
             {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : space['shape_1'],
                            'batch_size' : space["batch_size"],
                            'num_channels' : 1,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75}},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : space['shape_1'],
                            'kernel_shape' : space['kernel_shape_1'],
                            'num_channels_input' : 1,
                            'num_channels_output' : 60,
                            'batch_size' : space["batch_size"],
                            'act_enc' : 'rectifier',}},
                {'name' : 'Pool',
                    'params' : {'image_shape' : space['shape_2'],
                        'pool_shape' : space['pool_1'],
                        'num_channels' : 60}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [21, 21],
                            'batch_size' : space["batch_size"],
                            'num_channels' : 60,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75
                            }},
                {'name' : 'Convolution',
                    'params' : {'image_shape' : space['shape_3'],
                            'kernel_shape' : space['kernel_shape_2'],
                            'num_channels_input' : 1,
                            'num_channels_output' : 60,
                            'batch_size' : space["batch_size"],
                            'act_enc' : 'rectifier',}},
                {'name' : 'Pool',
                    'params' : {'image_shape' : space['shape_4'],
                        'pool_shape' : space['pool_2'],
                        'num_channels' : 60}},
                {'name' : 'LocalResponseNormalize',
                    'params' : {'image_shape' : [21, 21],
                            'batch_size' : space["batch_size"],
                            'num_channels' : 60,
                            'n' : 4,
                            'k' : 1,
                            'alpha' : 0e-04,
                            'beta' : 0.75
                            }}]

    space["mlp_act"] = "rectifier"
    space["mlp_input_corruption_levels"] = [None, None]
    space["mlp_hidden_corruption_levels"] = [0.5, 0.5]
    space["mlp_nunits"] = [1000]
    space["n_outs"] = 7
    space["bias_init"] = 0.1
    space["irange"] = 0.1
    space["random_filters"] = True

    run(space, max_evals, db = 'abc', exp_key = 'dd')
    return

def run(space, max_evals, db, exp_key):
    trials = MongoTrials('mongo://localhost:1234/{}/jobs'.format(db), exp_key = exp_key)
    print "Watiting for jobs workers to run jobs..."
    best = fmin(experiment, space, trials=trials, algo=tpe.suggest, max_evals= max_evals)
    print "best: {}".format(best)
    print "losses: {}".format(trials.losses())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'supervised trainer')
    parser.add_argument('-e', '--experiment', required = True)
    parser.add_argument('-m', '--max_evals', type = int, default = 1)
    args = parser.parse_args()

    if args.experiment == 'tfd_conv':
        tfd_conv(args.max_evals)


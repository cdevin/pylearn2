"""
pre-train the albedo model
"""

import os
import argparse
import numpy
import theano
import theano.tensor as T

import pylearn2
import pylearn2.config
from pylearn2.utils import serial
from utils.config import get_data_path, get_result_path

DATA_PATH = get_data_path()
RESULT_PATH = get_result_path()

# 1 layer Albedian model
train_1layer_yaml_string = """
!obj:pylearn2.train.Train {
    "dataset": !pkl: %(data_path)s,
    "model": !obj:noisy_encoder.models.naenc.NoisyAutoencoder {
        "nvis" : %(nvis)i,
        "nhid" : %(nhid)i,
        "act_enc": %(act_enc)s,
        "act_dec": %(act_dec)s,
        "input_corruptor": !obj:noisy_encoder.utils.corruptions.GaussianCorruptor {
            "stdev" : %(input_corruption_level)f, "avg": 0.5},
        "hidden_corruptor": !obj:noisy_encoder.utils.corruptions.BinomialCorruptorScaled {
            "corruption_level" : %(hidden_corruption_level)f}
    },
    "algorithm": !obj:pylearn2.training_algorithms.sgd.SGD {
        "learning_rate" : %(learning_rate)f,
        "batch_size" : %(batch_size)i,
        "monitoring_batches" : %(monitoring_batches)i,
        "monitoring_dataset" : !pkl: %(data_path)s,
        "init_momentum" : %(init_momentum)f,
        "cost" : !obj:pylearn2.costs.cost.SumOfCosts {
                "costs" : [!obj:pylearn2.costs.autoencoder.MeanBinaryCrossEntropy {},
                !obj:DLN.utils.costs.WeightsL1Cost {ratio: %(w_l1_ratio)d}]},
        "termination_criterion" : !obj:pylearn2.training_algorithms.sgd.EpochCounter {
            "max_epochs": %(max_epochs)i,
        },
    },
    "callbacks" : [!obj:pylearn2.training_algorithms.sgd.MonitorBasedLRAdjuster {
                    "shrink_amt": %(shrink_amt)f, grow_amt: %(grow_amt)f},
                    !obj:pylearn2.training_algorithms.sgd.MomentumAdjustor {
                    "final_momentum" : %(final_momentum)f,
                    "start" : %(start_momentum)f,
                    "saturate" : %(saturate_momentum)f}],
    "save_path": %(save_name)s,
    "save_freq": %(save_freq)i
}
"""

def experiment(state, channel):
    # update base yaml config with jobman commands
    yaml_string = state.yaml_string % (state)

    # generate .yaml file
    fname = 'invar.yaml'
    fp = open(fname, 'w')
    fp.write(yaml_string)
    fp.close()

    # now run yaml file with default train.py script
    train_obj = pylearn2.config.yaml_parse.load(open(fname,'r'))
    train_obj.main_loop()

    try:
        state.final_cost = float(train_obj.algorithm.monitor.channels['sgd_cost'].val_record[-1])
    except TypeError:
        pass

    return channel.COMPLETE

def train_1layer(submit = False):

    from jobman.tools import DD

    state = DD()

    state.data_path = os.path.join(DATA_PATH, "cifar10_local/pylearn2/train.pkl")
    state.nvis = 32*32*3
    state.nhid = 1000
    state.learning_rate = 0.001
    state.grow_amt = 1.001
    state.shrink_amt = 0.009
    state.init_momentum = 0.5
    state.final_momentum = 0.9
    state.start_momentum = 10
    state.saturate_momentum = 30
    state.w_l1_ratio = 0.0
    #state.act_enc = "!obj:DLN.models.dln.Rectifier {}"
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.input_corruption_level = 0.2
    state.hidden_corruption_level = 0.5
    state.batch_size = 20
    state.monitoring_batches = 2500
    state.normal_penalty = 1
    state.max_epochs = 300
    #state.save_name = os.path.join(RESULT_PATH, "naenc/mnist/l1_")
    state.save_name = os.path.join(RESULT_PATH, "naenc/cifar/l1_2_")
    state.save_freq = 1
    state.yaml_string = train_1layer_yaml_string

    experiment(state, None)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'noisy AE trainer')
    parser.add_argument('-t', '--task', choices = ['layer1'], required = True)
    args = parser.parse_args()

    if args.task == 'layer1':
        train_1layer()
    else:
        raise ValueError("Wrong task optipns {}".format(args.task))


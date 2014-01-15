import os
import gc
import argparse
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse

import ipdb

def layer_num_params(layer):

    params = layer.get_params()
    return sum(map(lambda x: x.get_value().size, params))

def extract(path, save_name):

    data = {'valid' : [], 'test' : [], 'train' : [],
            'epoch' : [], 'l0_params' : [], 'l2_params' : [],
            'l3_params' : [], 'l4_params' : [],
            'l2_pieces' : [], 'l2_kernel' : [],
            'l3_pieces' : [], 'l3_units' : [],
            'learning_rate' : [], 'momentum_saturate' : [],
            'final_momentum' : [], 'decay_factor' :[],
            'lr_saturate' : [], 'id' : []}
    fs = os.listdir(path)
    for f in fs:
        print f
        model = serial.load(os.path.join(path, f, 'best.pkl'))
        data['valid'].append(float(model.monitor.channels['valid_y_perplexity'].val_record[-1]))
        data['test'].append(float(model.monitor.channels['test_y_perplexity'].val_record[-1]))
        data['train'].append(float(model.monitor.channels['train_y_perplexity'].val_record[-1]))
        data['epoch'].append(model.monitor._epochs_seen)
        data['l0_params'].append(layer_num_params(model.layers[0]))
        data['l2_params'].append(layer_num_params(model.layers[2]))
        data['l3_params'].append(layer_num_params(model.layers[3]))
        data['l4_params'].append(layer_num_params(model.layers[4]))
        data['l2_pieces'].append(model.layers[2].num_pieces)
        data['l2_kernel'].append(model.layers[2].kernel_shape[0])
        data['l3_pieces'].append(model.layers[3].num_pieces)
        data['l3_units'].append(model.layers[3].num_units)

        gc.collect()

        conf = yaml_parse.load(open(os.path.join(path, f, 'model.yaml'), 'r'))
        data['learning_rate'].append(conf.algorithm.learning_rate.get_value())
        data['momentum_saturate'].append(conf.extensions[1].saturate)
        data['final_momentum'].append(conf.extensions[1].final_momentum)
        data['lr_saturate'].append(conf.extensions[2].saturate)
        data['decay_factor'].append(conf.extensions[2].decay_factor)
        data['id'].append(int(f))

    serial.save("{}.pkl".format(save_name), data)


def plot()

if __name__ == "__main__":


    parser  = argparse.ArgumentParser(description = 'Plot')
    parser.add_argument('-e', '--extract', default = False, action='store_true')
    parser.add_argument('-n', '--name', help = 'file name')
    parser.add_argument('-p', '--path')

    if args.extract:
        extract(args.path, args.name)
    else:
        plit(args.name)
    _, path, save_name = sys.argv

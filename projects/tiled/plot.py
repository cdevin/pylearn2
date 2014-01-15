import os
import sys
import gc
import numpy as np
from pylearn2.utils import serial

import ipdb

def layer_num_params(layer):

    params = layer.get_params()
    return sum(map(lambda x: x.get_value().size, params))

def extract(path, save_name):

    data = {'valid' : [], 'test' : [], 'train' : [],
            'epoch' : [], 'l0_params' : [], 'l2_params' : [],
            'l3_params' : [], 'l4_params' : [],
            'learning_rate' : [],
            'kernel_shape' : [],
            'id' : []}
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

        gc.collect()

        yaml = open(os.path.join(path, f, 'model.yaml'), 'r').read()
        data['learning_rate'].append(float(yaml[yaml.index('learning_rate'):].split(',')[0].split(':')[1]))
        data['kernel_shape'].append(int(yaml[yaml.index('kernel_shape'):].split(',')[0].split(':')[1].split('[')[1]))
        data['id'].append(int(f))


    serial.save("{}.pkl".format(save_name), data)


if __name__ == "__main__":
    _, path, save_name = sys.argv
    extract(path, save_name)

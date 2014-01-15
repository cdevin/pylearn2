import os
import gc
import argparse
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from matplotlib import pyplot as plt

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

def sort_index(my_list):
    return [i[0] for i in sorted(enumerate(my_list), key=lambda x:x[1])]

def plot(path):
    data = serial.load(path)

    # plot best
    plt.bar(data['id'], np.clip(data['test'], 0, 150))
    plt.title("test preplexity")
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.savefig('test_pre.png')
    plt.clf()

    # learning rate
    ind = sort_index(data['learning_rate'])
    plt.plot(np.asarray(data['learning_rate'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('learning rate')
    plt.ylabel('preplexity')
    plt.savefig('lr.png')
    plt.clf()

    # final momentum
    ind = sort_index(data['final_momentum'])
    plt.plot(np.asarray(data['final_momentum'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('final_momentum')
    plt.ylabel('preplexity')
    plt.savefig('final_momentum.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['momentum_saturate'])
    plt.plot(np.asarray(data['momentum_saturate'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('momentum_saturate')
    plt.ylabel('preplexity')
    plt.savefig('momentum_saturate.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['lr_saturate'])
    plt.plot(np.asarray(data['lr_saturate'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('lr_saturate')
    plt.ylabel('preplexity')
    plt.savefig('lr_saturate.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['decay_factor'])
    plt.plot(np.asarray(data['decay_factor'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('decay_factor')
    plt.ylabel('preplexity')
    plt.savefig('decay_factor.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['l0_params'])
    plt.plot(np.asarray(data['l0_params'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('l0_params')
    plt.ylabel('preplexity')
    plt.savefig('l0_params.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['l2_params'])
    plt.plot(np.asarray(data['l2_params'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('l2_params')
    plt.ylabel('preplexity')
    plt.savefig('l2_params.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['l3_params'])
    plt.plot(np.asarray(data['l3_params'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('l3_params')
    plt.ylabel('preplexity')
    plt.savefig('l3_params.png')
    plt.clf()

    # momentum saturate
    ind = sort_index(data['l4_params'])
    plt.plot(np.asarray(data['l4_params'])[ind], np.clip(data['test'], 0, 150)[ind])
    plt.xlabel('l4_params')
    plt.ylabel('preplexity')
    plt.savefig('l4_params.png')
    plt.clf()

    # momentum saturate
    plt.scatter(data['l2_kernel'], np.clip(data['test'], 0, 150))
    plt.xlabel('l2_kernel')
    plt.ylabel('preplexity')
    plt.savefig('l2_kernel.png')
    plt.clf()

    best_test = data['id'][np.argmin(data['test'])]
    print "Best test error job id: {}".format(best_test)

    #plt.show()
#

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'Plot')
    parser.add_argument('-e', '--extract', default = False, action='store_true')
    parser.add_argument('-n', '--name', help = 'file name')
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    if args.extract:
        extract(args.path, args.name)
    else:
        plot(args.name)

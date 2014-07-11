import os
import gc
import argparse
import numpy as np
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from matplotlib import pyplot as plt

def report(path, save_name):

    data = {'valid' : [], 'test' : [], 'train' : [], 'id' : []}
    fs = os.listdir(path)
    for f in fs:
        print f
        try:
            model = serial.load(os.path.join(path, f, 'best.pkl'))
        except:
            print "failed at {}".format(f)
            continue

        data['valid'].append(float(model.monitor.channels['valid_y_misclass'].val_record[-1]))
        data['test'].append(float(model.monitor.channels['test_y_misclass'].val_record[-1]))
        #data['train'].append(float(model.monitor.channels['train_y_perplexity'].val_record[-1]))

        gc.collect()

        conf = yaml_parse.load(open(os.path.join(path, f, 'model.yaml'), 'r'))
        data['id'].append(f)


    best_valid = data['id'][np.argmin(data['valid'])]
    best_test = data['test'][np.argmin(data['valid'])]
    print "Best job with lowest valid error id: {}, test error {}".format(best_valid, best_test)

if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'report')
    parser.add_argument('-t', '--task', choices=['report'], default = 'report')
    parser.add_argument('-n', '--name', help = 'file name')
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    if args.task == 'report':
        report(args.path, args.name)
    else:
        raise ValueError()


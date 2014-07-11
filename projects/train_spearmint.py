import os, shutil
import time
import argparse
import numpy
from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import preprocess
from pylearn2.utils import serial
from my_utils.config import get_data_path, get_result_path
from jobman.tools import DD




def experiment(state, channel):

    # udate path
    if channel is None:
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path += ''.join(alphabet[:7]) + '_'

    print "Saving results at: {}".format(state.save_path)
    # load and save yaml
    yaml_string = state.yaml_string % (state)
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    # now run yaml file with default train.py script
    start_time = time.time()
    train_obj = yaml_parse.load(yaml_string)
    train_obj.main_loop()
    state.run_time = time.time() - start_time

    ext = get_best_params_ext(train_obj.extensions)
    if ext != None:
        state.valid_score = float(ext.best_params['valid_y_misclass'])
        try:
            state.test_score = float(ext.best_params['test_y_misclass'])
        except KeyError:
            state.test_score = -1.
        print "Best valid: {}, best test: {}".format(state.valid_score, state.test_score)

    # save state
    serial.save(state.save_path + 'state.pkl', state)

    return state.valid_score, state.run_time



if __name__ == '__main__':
    with open('results.dat','r') as resfile:
        lines = resfile.readlines()
    newlines = []
    num_jobs = 0
    for line in lines:
        values = line.split()
        if len(values) < 3:
            continue
        val = values.pop(0)
        dur = values.pop(0)
        X = [float(values[0]), float(values[1])]
        print X
        if (val == 'P') and num_jobs ==0:
            val, dur = experiment(state)
            newlines.append(str(val) + str(dur)
                            + str(float(values[0])) + " "
                            + str(float(values[1])) + "\n")
            num_jobs += 1
        else:
            newlines.append(line)

    with open('results.dat','w') as outfile:
        for line in newlines:
            outfile.write(line)

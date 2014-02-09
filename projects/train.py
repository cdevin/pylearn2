import os, shutil
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
    try:
        yaml_string = state.yaml_string % (state)
    except ValueError:
        import ipdb
        ipdb.set_trace()
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    if state.db == 'SVHN':
        # transfer data to tmp
        orig_path = state.orig_path + 'h5/'
        tmp_path = state.data_path + 'h5/'
        train_f = 'splitted_train_32x32.h5'
        valid_f = 'valid_32x32.h5'
        test_f = 'test_32x32.h5'
        if any([not os.path.isfile(tmp_path + train_f), not os.path.isfile(tmp_path + valid_f), not os.path.isfile(tmp_path + test_f)]):
            print "Moving data to local tmp"
            if not os.path.isdir(tmp_path):
                os.makedirs(tmp_path)

            shutil.copy(orig_path + train_f, tmp_path)
            shutil.copy(orig_path + valid_f, tmp_path)
            shutil.copy(orig_path + test_f, tmp_path)

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)
    train_obj.main_loop()

    ext = get_best_params_ext(train_obj.extensions)
    if ext != None:
        state.valid_score = float(ext.best_params['valid_y_misclass'])
        try:
            state.test_score = float(ext.best_params['test_y_misclass'])
        except KeyError:
            state.test_score = -1.
        print "Best valid: {}, best test: {}".format(state.valid_score, state.test_score)

    return channel.COMPLETE


def experiment_finetune(state, channel, pre_trained):

    # udate path
    if channel is None:
        alphabet = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUWXYZ0123456789')
        numpy.random.shuffle(alphabet)
        state.save_path += ''.join(alphabet[:7]) + '_'

    # load and save yaml
    yaml_string = state.yaml_string % (state)
    with open(state.save_path + 'model.yaml', 'w') as fp:
        fp.write(yaml_string)

    # now run yaml file with default train.py script
    train_obj = yaml_parse.load(yaml_string)
    # load the pre-trained models
    model = serial.load(pre_trained)
    param_vals = model.get_param_values()
    import ipdb
    ipdb.set_trace()
    train_obj.model.layers[0].set_param_values([param_vals[2], param_vals[1]])
    #import ipdb
    #ipdb.set_trace()
    train_obj.main_loop()

    ext = get_best_params_ext(train_obj.extensions)
    if ext != None:
        state.valid_score = float(ext.best_params['valid_y_misclass'])
        try:
            state.test_score = float(ext.best_params['test_y_misclass'])
        except KeyError:
            state.test_score = -1.
        print "Best valid: {}, best test: {}".format(state.valid_score, state.test_score)

    return channel.COMPLETE


def get_best_params_ext(extensions):
    from noisy_encoder.utils.best_params import MonitorBasedBest
    for ext in extensions:
        if isinstance(ext, MonitorBasedBest):
             return ext



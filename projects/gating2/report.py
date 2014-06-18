import sys
import os
import glob
import numpy as np
from pylearn2.utils import serial
import ipdb


_, path = sys.argv

vals = []
tests = []

exps = os.listdir(path)
for exp in exps:
    try:
        model = glob.glob("{}/*best*pkl".format(os.path.join(path, exp)))[0]
    except IndexError:
        print "No model found in {}".format(exp)
        pass
    try:
        model = serial.load(model)
    except AttributeError:
        print "Failed on exp {}".format(exp)
        pass

    print "Experiment {}".format(exp)
    channel = model.monitor.channels['valid_perplexity']
    vals.append(channel.val_record[-1])
    print "Valid prplexity: {}".format(channel.val_record[-1])
    channel = model.monitor.channels['test_perplexity']
    tests.append(channel.val_record[-1])
    print "Test prplexity: {}".format(channel.val_record[-1])
    print

ind = np.argmin(vals)
print "Best exp was {} with val: {} and test: {}".format(exps[ind], vals[ind], tests[ind])


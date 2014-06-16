import sys
import os
import glob
from pylearn2.utils import serial
import ipdb


_, path = sys.argv

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
    print "Valid prplexity: {}".format(channel.val_record[-1])
    channel = model.monitor.channels['test_perplexity']
    print "Test prplexity: {}".format(channel.val_record[-1])
    print
    ipdb.set_trace()


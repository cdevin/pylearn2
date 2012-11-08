
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
import numpy as np
from pylearn2.datasets.tfd import TFD
from DLN.datasets.preprocessing import Scale
from DLN.config.config import get_data_path
from invarac.utils.funcs import lcn_numpy

print 'Loading TFD unsupervised dataset...'
unsupervised = TFD(which_set = 'unlabeled')
import ipdb

def lcn(data):

    topo = data.get_topological_view() / 255.
    res = lcn_numpy(topo.reshape(topo.shape[:3]))
    ipdb.set_trace()

lcn(unsupervised)
print "Preparing output directory..."
DATA_PATH = get_data_path()
data_dir = DATA_PATH + "/faces/TFD"
output_dir = data_dir + '/pylearn2'
serial.mkdir( output_dir )


pipeline = preprocessing.Pipeline()
pipeline.items.append(Scale(255.))
#pipeline.items.append(preprocessing.GlobalContrastNormalization())
unsupervised.apply_preprocessor(preprocessor = pipeline, can_fit = True)

print 'Saving the unsupervised data'
unsupervised.use_design_loc(output_dir+'/unlabeled.npy')
serial.save(output_dir + '/unlabeled.pkl', unsupervised)

# Train
print "Loading train data"
train = TFD(which_set = 'train')
print "Preprocessing the test data"
train.apply_preprocessor(preprocessor = pipeline, can_fit = False)
print "Saving the test data"
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir+'/train.pkl', train)

# Valid
print "Loading valid data"
valid = TFD(which_set = 'valid')
print "Preprocessing the test data"
valid.apply_preprocessor(preprocessor = pipeline, can_fit = False)
print "Saving the test data"
valid.use_design_loc(output_dir+'/valid.npy')
serial.save(output_dir+'/valid.pkl', valid)

# Test
print "Loading the test data"
test = TFD(which_set = 'test')
print "Preprocessing the test data"
test.apply_preprocessor(preprocessor = pipeline, can_fit = False)
print "Saving the test data"
test.use_design_loc(output_dir+'/test.npy')
serial.save(output_dir+'/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',pipeline)



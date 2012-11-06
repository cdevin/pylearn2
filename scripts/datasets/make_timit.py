"""
This script makes a dataset of TIMIT

"""
import numpy as np
from pylearn2.utils import serial
from utils.config import get_data_path
from noisy_encoder.datasets.timit import TIMIT

print 'Making train set...'
train = TIMIT('train', shuffle = True)

DATA_PATH = get_data_path()
data_dir = DATA_PATH + "timit"
output_dir = data_dir + '/pylearn2'
serial.mkdir( output_dir )

train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)


print "Making valid data"
valid = TIMIT('valid', shuffle = True)
valid.use_design_loc(output_dir + '/valid.npy')
serial.save(output_dir + '/valid.pkl', valid)



print "Making test data"
test = TIMIT('test', shuffle = True)
test.use_design_loc(output_dir + '/test.npy')
serial.save(output_dir + '/test.pkl', test)



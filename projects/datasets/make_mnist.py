"""
This script makes a dataset of MNIST images

"""

import numpy as np
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from my_utils.config import get_data_path

print 'Loading MNIST train set...'
train = MNIST('train', center = True)

print "Preparing output directory..."
DATA_PATH = get_data_path()
data_dir = DATA_PATH + "mnist"
output_dir = data_dir + '/pylearn2'
serial.mkdir( output_dir )

#pipeline = preprocessing.Pipeline()
##pipeline.items.append(preprocessing.GlobalContrastNormalization())
#unsupervised.apply_preprocessor(preprocessor = pipeline, can_fit = True)

print 'Saving the train data'
train.use_design_loc(output_dir+'/train.npy')
serial.save(output_dir + '/train.pkl', train)

# Train
print "Loading test data"
test = MNIST('test', center = True)
#test.apply_preprocessor(preprocessor = pipeline, can_fit = True)
test.use_design_loc(output_dir + '/test.npy')
serial.save(output_dir + '/test.pkl', test)

#serial.save(output_dir + '/preprocessor.pkl',pipeline)


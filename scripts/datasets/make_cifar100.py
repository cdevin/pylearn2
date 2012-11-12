from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar100 import CIFAR100
from utils.config import get_data_path
from DLN.datasets.preprocessing import Scale
import numpy

print 'Loading CIFAR_100 train set...'
train = CIFAR100(which_set="train")

DATA_PATH = get_data_path()
data_dir = DATA_PATH + "cifar100"
output_dir = data_dir + '/zca_512_d'
serial.mkdir( output_dir )

print "Pre-processin train set..."
pipeline = preprocessing.Pipeline()
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA(n_components=513, n_drop_components=1))
train.apply_preprocessor(preprocessor=pipeline, can_fit=True)

train.y = train.y_fine
train.use_design_loc(output_dir + '/train.npy')
serial.save(output_dir + '/train.pkl', train)
numpy.save(output_dir + '/train_y.npy', train.y)

# Test
print 'Loading CIFAR_100 test set...'
test = CIFAR100(which_set="test")
print "Pre-processin test set..."
test.apply_preprocessor(preprocessor=pipeline, can_fit=True)
test.y = test.y_fine
test.use_design_loc(output_dir + '/test.npy')
serial.save(output_dir + '/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',pipeline)
numpy.save(output_dir + '/test_y.npy', test.y)

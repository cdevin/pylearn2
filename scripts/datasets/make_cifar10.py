from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar10 import CIFAR10
from utils.config import get_data_path
from DLN.datasets.preprocessing import Scale

print 'Loading CIFAR_10 train set...'
train = CIFAR10(which_set="train", rescale = True, center = True)

print "Preparing output directory..."
DATA_PATH = get_data_path()
data_dir = DATA_PATH + "cifar10_local"
output_dir = data_dir + '/pylearn2'
serial.mkdir( output_dir )


pipeline = preprocessing.Pipeline()
#pipeline.items.append(
    #preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000)
#)
pipeline.items.append(preprocessing.GlobalContrastNormalization())
pipeline.items.append(preprocessing.ZCA())
#pipeline.items.append(preprocessing.Standardize())
#pipeline.items.append(preprocessing.RemapInterval([-10, 10], [0.0, 1.0]))
train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
train.use_design_loc(output_dir + '/train.npy')
serial.save(output_dir + '/train.pkl', train)

# Test
test = CIFAR10(which_set="test", rescale = True, center = True)
test.apply_preprocessor(preprocessor=pipeline, can_fit=True)
test.use_design_loc(output_dir + '/test.npy')
serial.save(output_dir + '/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',pipeline)

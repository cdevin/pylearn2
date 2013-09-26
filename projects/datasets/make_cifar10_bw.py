from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from utils.datasets.cifar10_bw import CIFAR10_BW
from utils.config import get_data_path

print 'Loading CIFAR_10 train set...'
train = CIFAR10_BW(which_set="train")

print "Preparing output directory..."
DATA_PATH = get_data_path()
data_dir = DATA_PATH + "cifar10_bw"
output_dir = data_dir + '/pylearn2_2'
serial.mkdir( output_dir )


pipeline = preprocessing.Pipeline()
#pipeline.items.append(
    #preprocessing.ExtractPatches(patch_shape=(8, 8), num_patches=150000)
#)
pipeline.items.append(preprocessing.GlobalContrastNormalization())
#pipeline.items.append(preprocessing.ZCA())

train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
train.use_design_loc(output_dir + '/train.npy')
serial.save(output_dir + '/train.pkl', train)

# Test
test = CIFAR10_BW(which_set="test")
test.apply_preprocessor(preprocessor=pipeline, can_fit=True)
test.use_design_loc(output_dir + '/test.npy')
serial.save(output_dir + '/test.pkl', test)

serial.save(output_dir + '/preprocessor.pkl',pipeline)

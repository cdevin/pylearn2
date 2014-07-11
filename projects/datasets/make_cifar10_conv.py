import argparse
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar10 import CIFAR10
from DLN.datasets.preprocessing import Scale
from my_utils.config import get_data_path

def make():

    train = CIFAR10(which_set="train", rescale = True, center = True, gcn = 55.)
    print train.X.max(), train.X.min()
    print "Preparing output directory..."
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "cifar10_conv"
    output_dir = data_dir + '/pylearn2'
    serial.mkdir( output_dir )

    pipeline = preprocessing.Pipeline()
    pipeline.items.append(preprocessing.RemapInterval([train.X.min(), train.X.max()], [0., 1.]))
    train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    train.use_design_loc(output_dir + '/train.npy')
    serial.save(output_dir + '/train.pkl', train)
    print train.X.max(), train.X.min()

    # Test
    test = CIFAR10(which_set="test", rescale = True, center = True, gcn = 55.)
    test.apply_preprocessor(preprocessor=pipeline, can_fit=True)
    test.use_design_loc(output_dir + '/test.npy')
    serial.save(output_dir + '/test.pkl', test)

    serial.save(output_dir + '/preprocessor.pkl',pipeline)


if __name__ == "__main__":
    make()

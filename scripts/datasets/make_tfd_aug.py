import numpy
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
import numpy as np
from pylearn2.datasets.tfd import TFD
from DLN.datasets.preprocessing import Scale
from DLN.config.config import get_data_path
from noisy_encoder.scripts.datasets.utils import reflect, shuffle, corner_shuffle

def make_data(which, fold, seed =2322):

    rng = numpy.random.RandomState(seed)

    print "Prcoessing {}...".format(which)
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "faces/TFD"
    output_dir = data_dir + '/pylearn2_aug/{}'.format(fold)
    serial.mkdir( output_dir )

    data= TFD(which_set = which, fold = fold, center = False)
    #data = apply_lcn(data)
    data.y = numpy.concatenate(data.y)

    if which == 'train':
        data.X, data.y = corner_shuffle(data.X, data.y, (data.X.shape[0], 48, 48), rng)
        data.X, data.y= reflect(data.X, data.y, (data.X.shape[0], 48, 48))
        data.X, data.y = shuffle(data.X, data.y, rng)

    data.X = data.X / 255.

    data.use_design_loc(output_dir + '/{}.npy'.format(which))
    serial.save(output_dir + '/{}.pkl'.format(which), data)


if __name__ == "__main__":
    make_data('train', 4)
    make_data('valid', 4)
    make_data('test', 4)

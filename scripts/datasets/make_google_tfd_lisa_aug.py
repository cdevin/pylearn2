import numpy
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
from DLN.datasets.preprocessing import Scale
from DLN.config.config import get_data_path
from noisy_encoder.datasets.lisa import Lisa
from noisy_encoder.datasets.google_tfd import GoogleTFD
from noisy_encoder.scripts.datasets.utils import reflect, shuffle, corner_shuffle

mapper = {'train' : 0, 'valid' : 1, 'test': 2}

def make_data(which, seed = 2322):

    assert which in mapper.keys()

    rng = numpy.random.RandomState(seed)

    print "Prcoessing {}...".format(which)
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "faces/google_tfd_lisa_aug"
    output_dir = data_dir + '/pylearn2'
    serial.mkdir( output_dir )

    if which == 'train':
        data= GoogleTFD(shuffle = True)
        # augumnet
        data.X, data.y = reflect(data.X, data.y, (data.X.shape[0], 48, 48))
        data.X, data.y = corner_shuffle(data.X, data.y, (data.X.shape[0], 48, 48), 3, rng)
        data.X, data.y = shuffle(data.X, data.y, rng)
    else:
        data = Lisa(which, shuffle = True, one_hot = True)

    data.X = data.X / 255.
    data.use_design_loc(output_dir + '/{}.npy'.format(which))
    serial.save(output_dir + '/{}.pkl'.format(which), data)


if __name__ == "__main__":
    make_data('train')
    make_data('valid')
    make_data('test')

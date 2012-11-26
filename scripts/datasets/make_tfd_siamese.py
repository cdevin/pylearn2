
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.utils import string_utils
import numpy as np
from pylearn2.datasets.tfd import TFD
from DLN.datasets.preprocessing import Scale
from DLN.config.config import get_data_path
#from invarac.utils.funcs import lcn_numpy
from lcn import lcn
from theano import tensor
import theano
import numpy
from copy import deepcopy
from noisy_encoder.scripts.datasets.utils import reflect, shuffle, corner_shuffle, apply_lcn


def make_data(which, fold, seed = 2322):

    rng = numpy.random.RandomState(seed)

    print "Prcoessing {}...".format(which)
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "faces/TFD"
    output_dir = data_dir + '/siamese/{}'.format(fold)
    serial.mkdir( output_dir )

    data= TFD(which_set = which, fold = fold, center = False)
    data.y = numpy.concatenate(data.y)
    data.y_identity = numpy.concatenate(data.y_identity)

    iden = numpy.unique(data.y_identity)
    new_x_emot = []
    new_x_neutral = []
    for item in iden:
        # find list of images with item idendity
        ind = numpy.where(data.y_identity == item)[0]
        y = data.y[ind]
        # check if we have neutral
        if numpy.any(data.y[ind] == 6):
            neutral = ind[numpy.where(data.y[ind] == 6)[0][0]]
            for emot in data.y[ind]:
                if emot != 6:
                    new_x_neutral.append(neutral)
                    new_x_emot.append(ind[numpy.where(data.y[ind] == emot)[0][0]])

    print "Numnber of new samples :{}/{}".format(len(new_x_emot),data.X.shape[0])

    data_neutral = deepcopy(data)
    data_neutral.X = data.X[new_x_neutral]
    data_neutral.y = data.y[new_x_neutral]
    data_neutral.y_identity = data.y_identity[new_x_neutral]

    data_emot = deepcopy(data)
    data_emot.X = data.X[new_x_emot]
    data_emot.y = data.y[new_x_emot]
    data_emot.y_identity = data.y_identity[new_x_emot]


    # augumentation
    if which == 'train':
        data_neutral.X, data_neutral.y = corner_shuffle(data_neutral.X, data_neutral.y, (data_neutral.X.shape[0], 48, 48), rng)
        data_neutral.X, data_neutral.y= reflect(data_neutral.X, data_neutral.y, (data_neutral.X.shape[0], 48, 48))
        data_neutral.X, data_neutral.y = shuffle(data_neutral.X, data_neutral.y, rng)

        data_emot.X, data_emot.y = corner_shuffle(data_emot.X, data_emot.y, (data_emot.X.shape[0], 48, 48), rng)
        data_emot.X, data_emot.y= reflect(data_emot.X, data_emot.y, (data_emot.X.shape[0], 48, 48))
        data_emot.X, data_emot.y = shuffle(data_emot.X, data_emot.y, rng)

    data_neutral.X = data_neutral.X / 255.
    data_emot.X = data_emot.X / 255.


    data_neutral.use_design_loc(output_dir + '/{}_neutral.npy'.format(which))
    serial.save(output_dir + '/{}_neutral.pkl'.format(which), data_neutral)
    data_emot.use_design_loc(output_dir + '/{}.npy'.format(which))
    serial.save(output_dir + '/{}.pkl'.format(which), data_emot)


if __name__ == "__main__":
    make_data('train', 4)
    make_data('valid', 4)
    make_data('test', 4)

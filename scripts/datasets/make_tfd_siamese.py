
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


def make_data(which, fold):

    print "Prcoessing {}...".format(which)
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "faces/TFD"
    output_dir = data_dir + '/siamese/{}'.format(fold)
    serial.mkdir( output_dir )

    data= TFD(which_set = which, fold = fold, center = False)
    #data = apply_lcn(data)
    data.X = data.X / 255.
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


    data_neutral.use_design_loc(output_dir + '/{}_neutral.npy'.format(which))
    serial.save(output_dir + '/{}_netural.pkl'.format(which), data_neutral)
    data_emot.use_design_loc(output_dir + '/{}_emot.npy'.format(which))
    serial.save(output_dir + '/{}_emot.pkl'.format(which), data_emot)


if __name__ == "__main__":
    make_data('train', 1)
    make_data('valid', 1)
    make_data('test', 1)

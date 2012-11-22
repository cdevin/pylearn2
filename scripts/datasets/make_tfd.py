
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


def apply_lcn(data):

    # TODO Clean it up, it's a mess
    x = tensor.matrix()
    f = theano.function(inputs=[x],outputs=lcn(x.reshape((1,48,48)),(48,48)))

    topo = data.get_topological_view() / 255.
    res = numpy.concatenate([f(item.reshape((48, 48))) for item in topo])
    data.X = res.reshape((topo.shape[0], topo.shape[1] * topo.shape[1]))
    return data


def make_data(which, fold):

    print "Prcoessing {}...".format(which)
    DATA_PATH = get_data_path()
    data_dir = DATA_PATH + "faces/TFD"
    output_dir = data_dir + '/pylearn2/{}'.format(fold)
    serial.mkdir( output_dir )

    data= TFD(which_set = which, fold = fold, center = False)
    #data = apply_lcn(data)
    data.X = data.X / 255.
    data.y = numpy.concatenate(data.y)

    data.use_design_loc(output_dir + '/{}.npy'.format(which))
    serial.save(output_dir + '/{}.pkl'.format(which), data)


if __name__ == "__main__":
    make_data('train', 3)
    make_data('valid', 3)
    make_data('test', 3)

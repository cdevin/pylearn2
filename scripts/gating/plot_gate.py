import numpy
from theano import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import theano.tensor as T
import matplotlib.pyplot as plt

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = 100
model.set_batch_size(batch_size)

assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()

test.X = test.X.astype('float32')
test.y = test.y.astype('float32')


Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
#yb = model.get_output_space().make_batch_theano()
#yb.name = 'yb'


gate_output = model.layers[0].gater.fprop(Xb)

f = function([Xb],gate_output)


# The averaging math assumes batches are all same size
assert test.X.shape[0] % batch_size == 0

def accs():
    assert isinstance(test.X.shape[0], (int, long))
    assert isinstance(batch_size, py_integer_types)
    rval = []
    for i in xrange(test.X.shape[0]/batch_size):
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        rval.append(f(x_arg))

    return numpy.concatenate(rval)


result = accs()


def plot_hist(x, name, nbins = 100):
    colors = ['r', 'b', 'g', 'c', 'y']
    for i in range(x.shape[1]):
        plt.hist(x[:,i], bins = nbins, color = colors[i])
    plt.savefig("{}.png".format(name))
    plt.clf()



test_y = numpy.argmax(test.y, axis=1)
for i in range(test.y.shape[1]):
    plot_hist(result[numpy.where(test_y == i)], name = i)

#plot_hist(result)

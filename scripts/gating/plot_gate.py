import numpy
from theano import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import theano.tensor as T
import matplotlib.pyplot as plt
from pylearn2.format.target_format import OneHotFormatter
from collections import Counter

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


gater = model.layers[0].gater
gate_output = gater.fprop(Xb)
#gate_output = T.argmax(gate_output, axis=1)
gate_output = OneHotFormatter(gater.layers[-1].n_classes).theano_expr(T.argmax(gate_output, axis=1))


gater2 = model.layers[1].gater
gate_output2 = gater2.fprop(model.layers[0].fprop(Xb))
gate_output2 = OneHotFormatter(gater2.layers[-1].n_classes).theano_expr(T.argmax(gate_output2, axis=1))

f = function([Xb],[gate_output, gate_output2])


# The averaging math assumes batches are all same size
assert test.X.shape[0] % batch_size == 0

def accs():
    assert isinstance(test.X.shape[0], (int, long))
    assert isinstance(batch_size, py_integer_types)
    rval = []
    rval2 = []
    for i in xrange(test.X.shape[0]/batch_size):
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        r1, r2 = f(x_arg)
        rval.append(r1)
        rval2.append(r2)

    return numpy.concatenate(rval), numpy.concatenate(rval2)


result, result2 = accs()


#def plot_hist(x, name, nbins = 100):
    #colors = ['r', 'b', 'g', 'c', 'y']
    #for i in range(x.shape[1]):
        #plt.hist(x[:,i], bins = nbins, color = colors[i])
    #plt.savefig("{}.png".format(name))
    #plt.clf()

def plot_hist(data, name):
    x = numpy.arange(data.shape[1])
    data_max = numpy.argmax(data, axis=1)
    y = [len(data_max[data_max == item]) for item in x]
    #import ipdb
    #ipdb.set_trace()
    print x, y
    #plt.bar(x, y)
    #plt.savefig("{}.png".format(name))
    #plt.clf()


test_y = numpy.argmax(test.y, axis=1)
for i in range(test.y.shape[1]):
    plot_hist(result[numpy.where(test_y == i)], name = i)


for i in range(test.y.shape[1]):
    plot_hist(result2[numpy.where(test_y == i)], name = i)

res_1d = numpy.argmax(result, axis=1)
print Counter(list(res_1d))
res_1d = numpy.argmax(result2, axis=1)
print Counter(list(res_1d))
import ipdb
ipdb.set_trace()
#plot_hist(result)

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


color_list = ['#CD0000', '#1E90FF', '#FFFF00', '#00EE00', '#FF34B3',
                '#63B8FF', '#FFF68F', '#8E8E38', '#00C78C', '#ff00ff', '#00ffff',
                '#ffff00', '#5528b2', '#3AEAA4', '#E4BFBA', '#3F9197', '#F83D17',
                '#30577B', '#5E7B30', '#5635E7', '#8575CD', '#3DC72D', '#C72D7F',
                '#3D0C20', '#1B31BF', '#20BF1B', '#133B12', '#3450AD', '#57689E',
                '#3B315E', '#F7CE52', '#CAC8DB', '#393654', '#080717', '#0AD1D1',
                '#7F3FC4', '#C43F93', '#A8B1E6', '#151C42', '#84A383', '#0FBD06']



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
if 'Sigmoid' in str(gater.layers[-1]):
    n_gates = gater.layers[-1].dim
else:
    n_gates = gater.layers[-1].n_classes
gate_output = OneHotFormatter(n_gates).theano_expr(T.argmax(gate_output, axis=1))


gater2 = model.layers[1].gater
if 'Sigmoid' in str(gater.layers[-1]):
    n_gates2 = gater2.layers[-1].dim
else:
    n_gates2 = gater2.layers[-1].n_classes
gate_output2 = gater2.fprop(model.layers[0].fprop(Xb))
gate_output2 = OneHotFormatter(n_gates2).theano_expr(T.argmax(gate_output2, axis=1))

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

def plot_hist(data, labels, name):

    test_y = numpy.argmax(labels, axis=1)
    #colors = ['r', 'b', 'g', 'c', 'y', 'm']
    width = .1
    for i in range(labels.shape[1]):
        col = data[numpy.where(test_y == i)]
        x = numpy.arange(col.shape[1])
        data_max = numpy.argmax(col, axis=1)
        y = [len(data_max[data_max == item]) for item in x]
        print y
        plt.bar(x, y, color=color_list[i])
        plt.xticks(x)
        plt.savefig("{}_{}.png".format(name, i))
        plt.clf()


#import ipdb
#ipdb.set_trace()
#test_y = numpy.argmax(test.y, axis=1)
#for i in range(test.y.shape[1]):
    #plot_hist(result[numpy.where(test_y == i)], name = i)
#
#for i in range(test.y.shape[1]):
    #plot_hist(result2[numpy.where(test_y == i)], name = i)


plot_hist(result, test.y, 'l1')
plot_hist(result2, test.y, 'l2')

res_1d = numpy.argmax(result, axis=1)
print Counter(list(res_1d))
res_1d = numpy.argmax(result2, axis=1)
print Counter(list(res_1d))
#import ipdb
#ipdb.set_trace()
#plot_hist(result)

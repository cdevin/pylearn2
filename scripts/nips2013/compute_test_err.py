
from theano import function
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys

_, model_path = sys.argv

model = serial.load(model_path)

src = model.dataset_yaml_src
batch_size = 100
model.set_batch_size(batch_size)

assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()
#assert test.X.shape[0] == 10000

test.X = test.X.astype('float32')
test.y = test.y.astype('float32')

import theano.tensor as T

Xb = model.get_input_space().make_batch_theano()
Xb.name = 'Xb'
yb = model.get_output_space().make_batch_theano()
yb.name = 'yb'

### get error
Y_hat = model.mf(Xb)[-1]
Y_hat.name = 'Y hat'
Y_hat_argmax = T.argmax(Y_hat, axis = 1)
yb_argmax = T.argmax(yb, axis = 1)
if Y_hat_argmax.dtype != Y_hat.dtype:
    argmax = T.cast(Y_hat_argmax, Y_hat.dtype)
mf1acc = 1.- T.neq(yb_argmax, Y_hat_argmax).mean()
batch_acc = function([Xb,yb],[mf1acc])

# The averaging math assumes batches are all same size
assert test.X.shape[0] % batch_size == 0

def accs():
    mf1_accs = []
    assert isinstance(test.X.shape[0], (int, long))
    assert isinstance(batch_size, py_integer_types)
    for i in xrange(test.X.shape[0]/batch_size):
        print i
        x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
        if Xb.ndim > 2:
            x_arg = test.get_topological_view(x_arg)
        mf1_accs.append( batch_acc(x_arg,
            test.y[i*batch_size:(i+1)*batch_size,:])[0])
    return sum(mf1_accs) / float(len(mf1_accs))


result = accs()


print 1. - result

from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
import numpy as np

out_path = sys.argv[1]
model_paths = sys.argv[2:]

class Committee(object):

    def set_batch_size(self, batch_size):
        for member in self.members:
            member.set_batch_size(batch_size)

    def get_input_space(self):
        return self.members[0].get_input_space()

    def get_output_space(self):
        return self.members[0].get_output_space()

    def __init__(self):
        self.members = []

    def fprop(self, state_below):
        states = [member.fprop(state_below) for member in self.members]
        state = sum(states)
        return state

model = Committee()

for model_path in model_paths:
    submodel = serial.load(model_path)
    model.members.append(submodel)

src = model.members[0].dataset_yaml_src
batch_size = 100
model.set_batch_size(batch_size)


assert src.find('train') != -1
test = yaml_parse.load(src)
test = test.get_test_set()
m = test.X.shape[0]
extra = batch_size - m % batch_size
assert (m + extra) % batch_size == 0
import numpy as np
if extra > 0:
    test.X = np.concatenate((test.X, np.zeros((extra, test.X.shape[1]),
    dtype=test.X.dtype)), axis=0)
assert test.X.shape[0] % batch_size == 0




import theano.tensor as T

X = model.get_input_space().make_batch_theano()
X.name = 'Xb'

ymf = model.fprop(X)
ymf.name = 'ymf'

from theano import function

yl = T.argmax(ymf,axis=1)

f = function([X],yl)


y = []

for i in xrange(test.X.shape[0] / batch_size):
    x_arg = test.X[i*batch_size:(i+1)*batch_size,:]
    if X.ndim > 2:
        x_arg = test.get_topological_view(x_arg)
    y.append(f(x_arg.astype(X.dtype)))

y = np.concatenate(y)
assert y.ndim == 1
assert y.shape[0] == test.X.shape[0]
# discard any zero-padding that was used to give the batches uniform size
y = y[:m]

out = open(out_path, 'w')
for i in xrange(y.shape[0]):
    out.write('%d.0\n' % (y[i] + 1))
out.close()



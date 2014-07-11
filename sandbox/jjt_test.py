import theano
from theano import tensor
import numpy
import ipdb
from theano.tensor.nnet import sigmoid

x = tensor.matrix()
w = tensor.matrix()
acts = tensor.dot(x, w)
h = sigmoid(acts)
act_g = tensor.grad(h.sum(), acts)
j = w * act_g.dimshuffle(0, 'x', 1)
jjt = tensor.dot(w.T, w) * act_g.dimshuffle(0, 'x', 1) ** 2.

f = theano.function([x, w], jjt)

res = f(numpy.random.random((10, 100)).astype(numpy.float32), numpy.random.random((100, 200)).astype(numpy.float32))
ipdb.set_trace()



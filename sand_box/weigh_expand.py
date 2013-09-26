import numpy
import theano
from theano import tensor


x = tensor.vector()
bi= tensor.vector()

def slice(bin, x1):
    s
    return x1[b]
xp, _  = theano.scan()
tensor.concatenate([x[ind * s : (ind+1):s] * bi[ind] for ind in tensor.arange(s)])

f = theano.function(inputs[x, bi, s], outputs = xp)

dx = numpy.ones(10)
bix = [0,1,0]
print f(dx, bix, 3)

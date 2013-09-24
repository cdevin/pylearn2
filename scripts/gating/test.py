import numpy
import theano
import theano.tensor
import theano.sparse
from theano import tensor
from pylearn2.utils import sharedX

b_dim = 5
v_dim = 10
h_dim = 20
rng = numpy.random.RandomState(232)

w = sharedX(rng.uniform(size=(v_dim, h_dim)))
b = sharedX(numpy.ones(h_dim))
x = tensor.matrix('x')
g = tensor.matrix('g')

x_val = rng.uniform(size=(b_dim, v_dim))
g_val = numpy.zeros((b_dim, h_dim))
x.tag.test_value = x_val
g.tag.test_value = g_val

g = theano.sparse.csr_from_dense(g)
#z = theano.sparse.basic.sampling_dot(x, w.T, g)
#b = theano.sparse.basic.mul_s_v(g, b)
#z = theano.sparse.basic.add_s_s(z, b)
z = theano.tensor.dot(x, w) + b
z = theano.sparse.mul_s_d(g, z)
z = theano.sparse.basic.structured_sigmoid(z)
z_sum = theano.sparse.sp_sum(z)
z_g = tensor.grad(z_sum, g)

f = theano.function([x,g], [z, z_sum])
f2 = theano.function([x, g], [z, z_g])



r_z, r_zs = f(x_val, g_val)
print r_z.shape, r_zs
r_z, r_zg = f2(x_val, g_val)
print r_zg


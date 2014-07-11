import numpy
import theano
from theano import tensor
from theano.tensor.nnet import softmax
import pylab

z = tensor.matrix()
w = tensor.matrix()
out = (softmax(z) * (tensor.log(softmax(z)) - tensor.log(softmax(w)))).sum(axis=1).mean()
out_std = (softmax(z) * (tensor.log(softmax(z)) - tensor.log(softmax(w)))).sum(axis=1).std()
f = theano.function([z, w], out)
f_std = theano.function([z,w], out_std)

def kl(p,q):
    p = numpy.load(p)
    q = numpy.load(q)
    return f(p, q)


def s_err(p,q):
    p = numpy.load(p)
    q = numpy.load(q)
    rval = f_std(p,q)
    return 1.96 * rval / numpy.sqrt(10000)

path = "/data/lisatmp/mirzamom/results/icml2013/boost/"
y_axis = []
y_axis.append(kl(path + '1_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '5_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '10_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '50_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '100_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '500_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '1000_g.npy', path + 'nodropout_g.npy'))


y_err = []
y_err.append(s_err(path + '1_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '5_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '10_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '50_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '100_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '500_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '1000_g.npy', path + 'nodropout_g.npy'))




x_axis = [1, 5, 10, 50, 100, 500, 1000]
pylab.errorbar(x_axis, y_axis, yerr = y_err, label = 'MaxOut')
## tanh

path = "/data/lisatmp/mirzamom/results/icml2013/boost/tanh_"
y_axis = []
y_axis.append(kl(path + '1_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '5_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '10_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '50_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '100_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '500_g.npy', path + 'nodropout_g.npy'))
y_axis.append(kl(path + '1000_g.npy', path + 'nodropout_g.npy'))


y_err = []
y_err.append(s_err(path + '1_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '5_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '10_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '50_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '100_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '500_g.npy', path + 'nodropout_g.npy'))
y_err.append(s_err(path + '1000_g.npy', path + 'nodropout_g.npy'))



pylab.errorbar(x_axis, y_axis, yerr = y_err, label = 'tanh')

pylab.xlabel('# samples')
pylab.ylabel('KL divergence')
pylab.xscale('log')
pylab.title('Averaging Effect on mnist')
pylab.legend()
pylab.show()

import numpy
from pylearn2.datasets.mnist import MNIST

def accs(data):
    y_pred = numpy.argmax(data, 1)
    y = numpy.argmax(test.y, 1)
    return numpy.sum(y_pred != y) / float(y.shape[0])

def softmax(data):
    data = data - data.max()
    return numpy.exp(data) / numpy.exp(data).sum(1)[:, numpy.newaxis]


def get_stata(path):
    data = numpy.load(path)
    each_softmax = [softmax(item) for item in data]
    each_accs = numpy.asarray([accs(item) for item in each_softmax])

    nums = [1, 5, 10, 50, 100, 500, 1000]
    for num in nums:
        print "num: {}, std: {}".format(num, each_accs[:num].std())

        sub = data[:num]
        sub = sub.mean(0)
        sub = softmax(sub)
        print "num: {}, error: {}".format(num, accs(sub))



test = MNIST('test', one_hot = 1)
tanh = '/data/lisatmp/mirzamom/results/icml2013/boost2/max_1000_g.npy'
get_stata(tanh)

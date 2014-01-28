import sys
import numpy as np
from theano import function
from theano import tensor as T
from pylearn2.utils import serial
from noisylearn.projects.tiled.penntree import PennTree
import ipdb



SEQ_LEN = 5
SEN_LEN = 50
NUM = 100

_, path = sys.argv
words = np.load('/data/lisa/data/PennTreebankCorpus/dictionaries.npz')['unique_words']
model = serial.load(path)

test = PennTree('test', SEQ_LEN)

x = model.get_input_space().make_batch_theano()
y = model.fprop(x)
y = T.argmax(y, axis=1)
f = function([x], y)


def gen(data):

    for i in xrange(SEN_LEN):
        rval = f(data[-SEQ_LEN:].astype('float32').reshape(1, SEQ_LEN))
        data = np.concatenate((data, rval))

    return data

def translate(data):
    rval = ""
    #ipdb.set_trace()
    for item in data:
        rval += " {}".format(words[item])
    return rval


indx = range(test.num_examples)
np.random.shuffle(indx)
for i in xrange(NUM):
    print translate(gen(test.X[indx[i]]))
    print


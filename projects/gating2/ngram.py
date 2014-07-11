import numpy as np
from pylearn2.utils import serial
from collections import Counter


def pentree():


    path = ("${PYLEARN2_DATA_PATH}/PennTreebankCorpus/" +
            "penntree_char_and_word.npz")
    data = serial.load(path)
    return  data['train_words']


def unigram(data):

    count = Counter(data)
    voca_len = float(len(data))
    rval = np.asarray([count[i] / voca_len for i in xrange(len(count.keys()))])

    return rval


def save(data, name):
    np.save(name, data)

if __name__ == "__main__":
    data = pentree()
    rval = unigram(data)
    save(rval, 'penntree_unigram.npy')

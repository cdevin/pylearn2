"""
This script is for One billion word dataset
The data currently are sorted in list of sentenses.
This script convert them to a single list, inserting
endo of sentense symbol at the end of each sentence.

The convention is to use num_words + 1 for end of sentence
and num_words + 2 for begining of setences
"""



import os
import numpy as np
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess


def get_num_words():
    words = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
        "smt/billion/en/newsxx.stream_word_indxs.pkl"))
    words = np.asarray(words.values())
    return words.max()


def flatten(data, sentence_end):
    rval = []
    for item in data:
        rval.extend(item)
        rval.extend([sentence_end])

    return rval


def flatten_memap(data, sentece_end):
    size = 0
    print 'h'
    for item in data:
        size += len(item) + 1

    output = np.memmap('train.dat', mode='w+', shape=(size))
    ind = 0
    print 'f'
    for item in data:
        output[ind: ind + len(item)] = item
        ind += len(item)
        output[ind:ind+1] = sentence_end
        ind +1


def load_data(which_set):
    if which_set == 'test':
        return serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
            "smt/billion/en/newsxx.test.npy"))
    elif which_set == 'train':
        path = preprocess(os.path.join("${PYLEARN2_DATA_PATH}",
            "smt/billion/en/newsxx.npy"))
        return np.load(path)
    elif which_set == 'valid':
        # TODO Check if valid is sharing data with test set
        path = preprocess(os.path.join("${PYLEARN2_DATA_PATH}",
            "smt/billion/en/newsxx.heldout.npy"))
        return np.load(path)

if __name__ == "__main__":
    sentence_end = get_num_words() + 1

    # train
    #data = load_data('train')
    #flatten_memap(data, sentence_end)

    #print aaa
    for set_ in ['valid']:
        data = load_data(set_)
        flattened = flatten(data, sentence_end)
        np.save(set_, np.asarray(flattened) )


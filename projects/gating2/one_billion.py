"""
Steps to follow to construct the dataset

1-Extract words from the text files and construct the vocabulary:
    python one_billion.py -t vocabulary

2-Cleaun up the vocabulary:
    python one_billion.py -t clean

3-Make numpy array dataset:
    python one_billion.py -t dataset

"""



import sys
import os
import argparse
import numpy as np
from pylearn2.utils import serial
import ipdb


def construct_voc(files, voc = None, unigram = None):
    """
    Consturuct the vocabulary from text files

    Paramters
    ---------
        file: list
            list of text files
        voc: dict, optional
            dictinary of all the words, keys are the words, values indexes
        unigram: dict, optional
            dictionary of count of each word. Keys are indexes, values counts
    """

    if voc is None or unigram is None:
        voc = {'<S>' : 0, '</S>' : 1, '<UNK>' : 2}
        unigram = {}
        ind = 3
    else:
        ind = max(voc.keys())
    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file.readlines():
                words = line.rstrip('/n').split(' ')
                for item in words:
                    item = item.lower()
                    if item in voc.keys():
                        unigram[voc[item]] += 1
                    else:
                        voc[item] = ind
                        ind +=1
                        unigram[voc[item]] = 1
                        print ind


    return voc, unigram


def cleanup(voc, unigram):
    """
    Removes all words with count below 3, and re-assign indexes
    """

    print len(voc.keys())
    del voc['<S>']
    del voc['</S>']
    del voc['<UNK>']

    for item in voc.keys():
        if unigram[voc[item]] < 3:
            del voc[item]


    voc_ = {'<S>' : 0, '</S>' : 1, '<UNK>' : 2}
    unigram_ = {}
    ind = 3
    for key in voc.keys():
        voc_[key] = ind
        unigram_[ind] = unigram[voc[key]]
        ind += 1

    print len(voc_.keys())
    return voc_, unigram_


def make_dataset(voc, files):
    """
    Construct the dataset

    Paramters
    ---------
        voc: dict
            Vocabulary dictionary
        files:
            Dataset text files
    """

    data = []
    sent_ends = []
    ind = 0

    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file.readlines():
                words = line.rstrip('/n').split(' ')
                for item in words:
                    item = item.lower()
                    try:
                        key = voc[item]
                    except KeyError:
                        key = voc['<UNK>']
                    data.append(key)
                    ind += 1
                # end of sentence
                data.append(voc['</S>'])
                sent_ends.append(ind)

    return np.asarray(data, dtype = 'int64'), np.asarray(sent_ends, dtype = 'int64')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices = ['vocabulary', 'clean', 'dataset'])
    args = parser.parse_args()

    path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-" +\
            "modeling-benchmark-r13output/training-monolingual." +\
            "tokenized.shuffled/"
    print args.task
    if args.task == 'vocabulary':
        files = os.listdir(path)
        files = [os.path.join(path, item) for item in files]
        voc, unigram = construct_voc(files)
        serial.save('one_billion_voc_full.pkl', voc)
        serial.save('one_billion_unigram_full.pkl', unigram)
    elif args.task == 'clean':
        voc = serial.load('one_billion_voc_full.pkl')
        unigram = serial.load('one_billion_unigram_full.pkl')
        voc, unigram = cleanup(voc, unigram)
        serial.save('one_billion_voc.pkl', voc)
        serial.save('one_billion_unigram.pkl', unigram)
    elif args.task == 'dataset':
        files = os.listdir(path)
        files = [os.path.join(path, item) for item in files]
        voc = serial.load('one_billion_voc.pkl')
        data, sent_ends = make_dataset(voc, files)
        np.save('one_billion_train.npy', data)
        np.save('one_billion_train_sentence_end.npy', sent_ends)
    else:
        raise ValueError("Unknown task : {}".format(args.task))


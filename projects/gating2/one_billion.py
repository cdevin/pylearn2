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
import glob
import argparse
import numpy as np
from collections import Counter
from pylearn2.utils import serial
from noisylearn.utils.cache import CachedAttribute
from noisylearn.projects.gating2.parse_text import make_dataset
import ipdb

#class OneBillionWord(SequenceDataset):


    #def __init__(self, which_set, seq_len):


        #data = serial.load('dss.npy')
        #X = data
        #y = None
        #self.seq_len = seq_len

        ## get voc
        #voc = serial.load('')
        #self.num_words = len(voc.key())
        #self.end_sentence = voc['</S>']
        #self.begin_setence = voc['<S>']
        #del voc

        #super(OneBillionWord, self).__init__(X = X, y = y)

    #@CachedAttribute
    #def num_words(self):
        #voc = serial.load('')
        #return len(voc.keys())



def construct_voc(files):

    counter = Counter()
    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file:
                line = line.rstrip('/n')
                #counter.update(word.lower() for word in line.split())
                counter.update(word for word in line.split())


    return counter


def cleanup(counter):

    voc = {'<S>' : 0, '</S>' : 1, '<UNK>' : 2}
    ind = 3
    for item in counter:
        if counter[item] > 2:
            voc[item] = ind
            ind += 1

    return voc

def make_dataset2(voc, files):
    """
    Construct the dataset

    Paramters
    ---------
        voc: dict
            Vocabulary dictionary
        files:
            Dataset text files
    """

    data = np.array([], dtype = 'int64')
    sent_ends = np.array([], dtype = 'int64')
    ind = 0

    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file.readlines():
                words = line.rstrip('/n').split(' ')
                for item in words:
                    try:
                        key = voc[item]
                    except KeyError:
                        key = voc['<UNK>']
                    np.append(data, key)
                    ind += 1
                # end of sentence
                np.append(data, voc['</S>'])
                np.append(sent_ends, ind)
        ipdb.set_trace()

    return data, sent_ends


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices = ['vocabulary', 'clean', 'train_set', 'test_set'])
    args = parser.parse_args()

    train_path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-" +\
            "modeling-benchmark-r13output/training-monolingual." +\
            "tokenized.shuffled/"

    test_path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-" +\
            "modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"

    if args.task == 'vocabulary':
        files = os.listdir(train_path)
        files = [os.path.join(train_path, item) for item in files]
        counter = construct_voc(files)
        serial.save('one_billion_counter_full.pkl', counter)
    elif args.task == 'clean':
        counter = serial.load('one_billion_counter_full.pkl')
        voc = cleanup(counter)
        serial.save('one_billionr_voc.pkl', voc)
    elif args.task == 'train_set':
        files = os.listdir(train_path)
        files = [os.path.join(train_path, item) for item in files]
        voc = serial.load('one_billion_voc.pkl')
        #data, sent_ends = make_dataset(voc, files)
        data = make_dataset(voc, files)
        np.save('one_billion_train.npy', data)
        #np.save('one_billion_train_sentence_end.npy', sent_ends)
    elif args.task == 'test_set':
        files = glob.glob(test_path + 'news.en.heldout*')
        files = [os.path.join(test_path, item) for item in files]
        voc = serial.load('one_billion_voc.pkl')
        #data, sent_ends = make_dataset(voc, files)
        data = make_dataset(voc, files)
        np.save('one_billion_test.npy', data)
        #np.save('one_billion_test_sentence_end.npy', sent_ends)

    else:
        raise ValueError("Unknown task : {}".format(args.task))


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
from noisylearn.projects.gating2.dataset import SequenceDataset
from pylearn2.space import VectorSpace, CompositeSpace

class OneBillionWord(SequenceDataset):
    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len):

        if which_set not in self.valid_set_names:
            raise ValueError("which_set should have one of these values: {}".format(self.valid_set_names))
        
        if which_set =='train':
            data = serial.load('test/one_billion_train.npy')
        self.X = data.reshape((data.shape[0],1))
        self.y = None
        self.seq_len = seq_len

        x_space = VectorSpace(dim = seq_len)
        y_space = VectorSpace(dim=1)
        space = CompositeSpace((x_space, y_space))
        source = ('features', 'targets')
        self.data_specs = (space, source)
        self.X_space = x_space


        # get voc
        voc = serial.load('test/one_billionr_voc.pkl')
        self.num_words = len(voc)
        self.end_sentence = voc['</S>']
        self.begin_sentence = voc['<S>']
        del voc
        super(OneBillionWord, self).__init__(X = self.X, y = self.y)

    #@CachedAttribute
    def num_words(self):
        return self.num_words



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
    ind = 0

    for file in files:
        print "Processing {}".format(file)
        with open(file, 'r') as file:
            for line in file.readlines():
                words = line.rstrip('\n').split(' ')
                for item in words:
                    try:
                        key = voc[item]
                    except KeyError:
                        key = voc['<UNK>']
                    data = np.append(data, key)
                    ind += 1
                # end of sentence
                data = np.append(data, voc['</S>'])

    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices = ['vocabulary', 'clean', 'train_set', 'test_set'])
    args = parser.parse_args()

    # train_path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-" +\
    #         "modeling-benchmark-r13output/training-monolingual." +\
    #         "tokenized.shuffled/"
    train_path = "/u/huilgolr/data/1billion/train/"

    test_path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-" +\
            "modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/"

    if args.task == 'vocabulary':
        files = os.listdir(train_path)
        files = [os.path.join(train_path, item) for item in files]
        counter = construct_voc(files)
        serial.save('test/one_billion_counter_full.pkl', counter)
    elif args.task == 'clean':
        counter = serial.load('test/one_billion_counter_full.pkl')
        voc = cleanup(counter)
        print voc
        #ipdb.set_trace()
        serial.save('test/one_billionr_voc.pkl', voc)
    elif args.task == 'train_set':
        files = os.listdir(train_path)
        files = [os.path.join(train_path, item) for item in files]
        voc = serial.load('test/one_billionr_voc.pkl')

        data = make_dataset2(voc, files)
        np.save('test/one_billion_train.npy', data)
    elif args.task == 'test_set':
        files = glob.glob(test_path + 'news.en.heldout*')
        files = [os.path.join(test_path, item) for item in files]
        voc = serial.load('test/one_billion_voc.pkl')
        #data, sent_ends = make_dataset(voc, files)
        data = make_dataset(voc, files)
        np.save('test/one_billion_test.npy', data)
        #np.save('one_billion_test_sentence_end.npy', sent_ends)

    else:
        raise ValueError("Unknown task : {}".format(args.task))


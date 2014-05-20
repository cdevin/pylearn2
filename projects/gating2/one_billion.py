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
        voc = {'<S>' : 0, '</S>' : 1}
        unigram = {}
        ind = 2
    else:
        ind = max(voc.keys())
    for file in files:
        print "Procssing {}".format(file)
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


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', choices = ['vocabulary'])
    args = parser.parse_args()

    if args.task == 'vocabulary':
        path = "/data/lisatmp2/mirzamom/data/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/"
        files = os.listdir(path)
        files = [os.path.join(path, item) for item in files]
        voc, unigram = construct_voc(files)
        serial.save('one_billion_voc.pkl', voc)
        serial.save('one_billion_unigram.pkl', voc)
    else:
        raise ValueError("Wrong argument")


import cPickle
import argparse

import numpy as np

from charModel import CharModel
from wordModel import WordModel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chars",  default="/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl", help="")
    parser.add_argument("--words",  default="/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl", help="")
    parser.add_argument("--vocab",  default="/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl", help="")
    parser.add_argument("--model", default="", help="pickled model")
    parser.add_argument("--test", default="", help="test vocabulary")
    parser.add_argument("--character-based", action="store_true",
        help="character-based embedding")
    parser.add_argument("embeddings", help="embedding path")
    return parser.parse_args()

def main():
    model_path = '../pkls/schwenkRealSkipgram300_NoAda.pkl' 
    chars_path = '/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl'
    vocab_path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl'
    words_path = '/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl'
    embeddings_path = '/data/lisatmp3/devincol/embeddings/skipgram300.pkl'

    args = parse_args()

    print "Loading Data"
    with open(args.vocab) as f:
       vocab = cPickle.load(f)
    ivocab = {v:k for k,v in vocab.iteritems()}
       
    with open(args.words) as f:
       words = cPickle.load(f)

    with open(args.chars) as f:
       char_dict = cPickle.load(f)
    inv_dict = {v:k for k,v in char_dict.items()}
    inv_dict[0] = inv_dict[len(inv_dict.keys())-1]
    unknown =  inv_dict[0]

    if args.model != "":
        with open(args.model) as f:
           pylearn2_model = cPickle.load(f)
        embeddings = pylearn2_model.layers[0].get_params()[0].get_value() 
        np.save(args.embeddings, embeddings)
    else:
        embeddings = np.load(args.embeddings)
    print "embeddings", len(embeddings), embeddings[0].shape
    # with open(embeddings_path) as f:
    #    embeddings_path

    print "Building Model"
    # Change this to WordModel or CharModel depending on whther you are using word or character -based embeddings
    if args.character_based:
        model = CharModel(pylearn2_model, char_dict, embeddings=embeddings, fprop=pylearn2_model.layers[0].fprop, words=words)
    else:
        model = WordModel(None, vocab, embeddings)

    print "Calculating Closest Words"
    if __name__ == "__main__":
        if args.test != "":
            map(model.displayStringRun, args.test.split(','))
            #map(model.displayStringRun, ['cat', 'dog', 'France', 'france', 'Canada', 'Paris', 'paris', 'brother', 'mother',
            #                             'sister', 'dad', 'mom', 'pharmacy', 'farm', 'quite', 'quiet', 'quit', 'like',
            #                             'love', 'city', 'town'])
            #map(model.displayStringRun, ['monarch', 'democracy', 'political', 'raspberry', 'blueberry', 'accomplishment', 'applying', 'application'])
        else:
            map(model.displayIndexRun, range(100, 150))

if __name__ == "__main__":
    main()

"""
Pylearn2 wrapper for h5-format datasets of sentences. Dataset generates
ngrams and swaps 2 adjacent words. Targets are n-1 vectors indicating where 
swap happened. 
"""
__authors__ = ["Coline Devin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Coline Devin", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Coline Devin"
__email__ = "devincol@iro"


import os.path
import functools
import numpy
import tables
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from pylearn2.sandbox.nlp.datasets.shuffle2 import H5Shuffle
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Skipgram(H5Shuffle):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length,
                 start=0, stop=None, X_labels=None, _iter_num_batches=10000,
                 rng=_default_seed, load_to_memory=False):
        """
        Parameters
        ----------
        path : str
            The base path to the data
        node: str
            The node in the h5 file
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of words contained in a frame
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        super(H5Skipgram, self).__init__(path, node, which_set, frame_length, start=start,
                                         stop=stop, X_labels=X_labels, 
                                         _iter_num_batches=_iter_num_batches, rng=rng,
                                         load_to_memory=load_to_memory)

        features_space = IndexSpace(
            dim=1,
            max_labels=self.X_labels
        )
        features_source = 'features'

        targets_space = [
            IndexSpace(dim=1, max_labels=self.X_labels),
            IndexSpace(dim=1, max_labels=self.X_labels),
            IndexSpace(dim=1, max_labels=self.X_labels),
            IndexSpace(dim=1, max_labels=self.X_labels),
            IndexSpace(dim=1, max_labels=self.X_labels),
            IndexSpace(dim=1, max_labels=self.X_labels)]
        targets_source = tuple('target'+str(i) for i in range(len(targets_space)))
        self._spaces = [features_space] + targets_space
        space = CompositeSpace(self._spaces)
        self._sources = (features_source,)+ targets_source

        self.data_specs = (space, self._sources)

        def getFeatures(indexes):
            """
            .. todo::
                Write me
            """
            sequences = [self.node[i] for i in indexes]

            # Get random start point for ngram
            wis = [numpy.random.randint(0, len(s)-self.frame_length+1, 1)[0] for s in sequences]
            # end = min(len(s), self.frame_length+wi)
            # diff = max(self.frame_length +wi - len(s), 0)
            # x = s[wi:end] + [0]*diff

            # X = numpy.asarray([numpy.concatenate((s[wi:(min(len(s), self.frame_length+wi))],
            #      [0]*(max(self.frame_length +wi - len(s), 0)))) for s, wi in 
            #      zip(sequences, wis)])

            ngrams = numpy.asarray([s[wi:self.frame_length+wi] for s, wi in zip(sequences, wis)])

            # Words mapped to integers greater than input max are set to 1 (unknown)
            ngrams[ngrams>=self.X_labels] = 1
            middle = int(frame_length/2)
            X = numpy.transpose(ngrams[:,middle][numpy.newaxis])
            y = numpy.concatenate((ngrams[:,range(middle)], ngrams[:,range(middle+1,self.frame_length)]), axis=1)
            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            return X

        def getTarget(source_index, indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                y = numpy.transpose(self.lastY[0][:,source_index][numpy.newaxis])
                return y
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None

        targetFNs = [lambda indexes: getTarget(0, indexes), lambda indexes: getTarget(1, indexes),
                     lambda indexes: getTarget(2, indexes),lambda indexes: getTarget(3, indexes),
                     lambda indexes: getTarget(4, indexes), lambda indexes: getTarget(5, indexes)]
        # targetFNs = [(lambda indexes: getTarget(i, indexes)) for i in range(len(targets_space))]
        self.sourceFNs = {'target'+str(i): targetFNs[i] for i in range(len(targets_space))}
        print "sourceFNs", self.sourceFNs
        self.sourceFNs['features'] =  getFeatures
        


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
import random
import tables
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.sandbox.nlp.datasets.wmt14 import H5_WMT14
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip, safe_izip
from pylearn2.utils.iteration import FiniteDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class Skipgram(H5_WMT14):
    """
    A dataset class to obtain data examples of (word, context-word) from an h5
    structured as an array of sentence arrays.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, context_distance,
                 start=0, stop=None, X_labels=None,
		 _iter_num_batches=None, rng=_default_seed, 
                 cache_size=None, cache_delta=None):
        """
        Parameters
        ----------
        path : str
            The base path to the data
        node: str
            The node in the h5 file
        context_distance : int
            The possible distance of the context. Currently using uniform distance.
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Required if using caching.
            If not provided and not using caching, defaults to using whole file.
        X_labels : int
            The maximum label to allow in the dataset.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
	_iter_num_batches : int, optional
	    Determines number of batches to cycle through in one epoch. Used to
	    calculate the validation score without having to cycle through the
	    entire dataset. Defaults to 1000.
        cache_size : int, optional
            If cache_size is set, the dataset will initially only load the 
            first cache_size examples from the data. Making this larger will
            increase the possible distance between examples (because data is
            loaded sequentially)
        cache_delta : int, optional
            Required if cache_size is set. Every cache_delta examples 
            (approximately because it doesn't need to be a multiple of batches)
            the dataset will load an additional cache_delta examples from the
            data (in consecutive order). Making this larger will allow more 
            non-sequentiality, but if cache_delta is equal to cache_size,
            then there will be no overlap between caches.
        """
        super(Skipgram, self).__init__(path, node,
                 start=start, stop=stop, X_labels=X_labels,
		 _iter_num_batches=_iter_num_batches, rng=rng, 
                 cache_size=cache_size, cache_delta=cache_delta)
        self.context_distance = context_distance

        features_space = IndexSpace(
            dim=1,
            max_labels=self.X_labels
        )
        features_source = 'features'

        targets_space = IndexSpace(dim=1, max_labels=self.X_labels)

        targets_source = 'targets'

        spaces = [features_space, targets_space]
        space = CompositeSpace(spaces)
        source = (features_source, targets_source)
        self.data_specs = (space, source)

        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))


        def getFeatures(indexes):
            """
            .. todo::
            Write me
            """
            sequences = [self.samples_sequences[i] for i in indexes]
            # Get random source word index for "ngram"
            source_i = [numpy.random.randint(self.context_distance , 
                                             len(s)-self.context_distance, 1)[0] 
                        for s in sequences]
                            
            # Mikolov's implementation picks a random frame length for each input word and
            # then uses all words within this frame as outputs. To simulate this with only
            # one target word per input, we pick a random frame length and then pick a random
            # context word within that frame.
            frame_lengths = numpy.random.random_integers(1, self.context_distance, len(sequences))
            distances = [(random.choice([1, -1]) * 
                          numpy.random.random_integers(1, f)) for f in frame_lengths]

            source_i = [numpy.random.randint(f , len(s)-self.context_distance, 1)[0] 
                        for s, f in safe_izip(sequences, frame_lengths)]

            target_i = [min(max(s_i + d_i, 0), len(s)-1)
                        for s_i, d_i, s in safe_izip(source_i, distances, sequences)]
                        
            # Words mapped to integers greater than input max are set to 1 (unknown)
            X = numpy.asarray([numpy.asarray([s[i]]) for i, s in safe_izip(source_i, sequences)])
            over_v = numpy.where(X.T >= self.X_labels)[1]
            X[over_v] = numpy.asarray([1])
            y = numpy.asarray([numpy.asarray([s[i]]) for i, s in safe_izip(target_i, sequences)])
            over_v = numpy.where(y.T >= self.X_labels)[1]
            y[over_v] = numpy.asarray([1])
            # Store the targets generated by these indices.
            self.lastY = y
            #print X
            #print y
            return X

        def getTarget(indexes):
            return self.lastY

        self.sourceFNs['features'] =  getFeatures
        self.sourceFNs['targets'] = getTarget

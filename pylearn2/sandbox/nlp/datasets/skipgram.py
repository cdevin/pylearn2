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
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.sandbox.nlp.datasets.shuffle2 import H5Shuffle
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip, safe_izip
from pylearn2.utils.iteration import FiniteDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Skipgram(H5Shuffle):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length,
                 start=0, stop=None, X_labels=None,
		 _iter_num_batches=None, rng=_default_seed, 
                 load_to_memory=False, cache_size=None,
                 cache_delta=None, schwenk=False):
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
        super(H5Skipgram, self).__init__(path, node, which_set, frame_length,
                 start=start, stop=stop, X_labels=X_labels,
		 _iter_num_batches=_iter_num_batches, rng=rng, 
                 load_to_memory=load_to_memory, cache_size=cache_size,
                 cache_delta=cache_delta, schwenk=schwenk)

        features_space = IndexSpace(
            dim=1,
            max_labels=self.X_labels
        )
        features_source = 'features'

        targets_space = IndexSpace(dim=1, max_labels=self.X_labels)

        targets_source = 'targets'

        spaces = [features_space, targets_space]
        print "Space len", len(spaces)
        space = CompositeSpace(spaces)
        source = (features_source, targets_source)
        print "source len", len(source)
        self.data_specs = (space, source)
        print self.data_specs

        def getFeatures(indexes):
            """
            .. todo::
                Write me
            """
            if self._load_to_memory:
                sequences = [self.samples_sequences[i] for i in indexes]
            else:
                sequences = [self.node[i] for i in indexes]
            # Get random source word index for "ngram"
            source_i = [numpy.random.randint(self.frame_length/2 +1, len(s)-self.frame_length/2, 1)[0] 
                        for s in sequences]
            target_i = [min(abs(int(numpy.random.normal(s_i, self.frame_length/3.0))), len(s)-1)
                        for s_i, s in safe_izip(source_i, sequences)]
            

            # Words mapped to integers greater than input max are set to 1 (unknown)
            X = [numpy.asarray([s[i]]) for i, s in safe_izip(source_i, sequences)]
            X[X>=self.X_labels] = numpy.asarray([1])
            X = numpy.asarray(X)
            y = [numpy.asarray([s[i]]) for i, s in safe_izip(target_i, sequences)]
            y[y>=self.X_labels] = numpy.asarray([1])
            y = numpy.asarray(y)
            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            #print X
            #print y
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                #y = numpy.transpose(self.lastY[0][:,source_index][numpy.newaxis])
                #print y
                #print y[-1]
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None

        # targetFNs = [
        #     lambda indexes: getTarget(0, indexes), lambda indexes: getTarget(1, indexes),
        #     lambda indexes: getTarget(2, indexes),lambda indexes: getTarget(3, indexes),
        #     lambda indexes: getTarget(4, indexes), 
        #     lambda indexes: getTarget(5, indexes)]
        # targetFNs = [(lambda indexes: getTarget(i, indexes)) for i in range(len(targets_space))]
        #self.sourceFNs = {'target'+str(i): targetFNs[i] for i in range(len(targets_space))}
        #print "sourceFNs", self.sourceFNs
        self.sourceFNs['features'] =  getFeatures
        self.sourceFNs['targets'] = getTarget


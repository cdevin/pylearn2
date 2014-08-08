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
from multiprocessing import Process, Queue

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class BilingSkipgram(H5Shuffle):
    """
    Writeme
    """
    _default_seed = (17, 2, 946)

    def __init__(self, source_path, target_path, which_set, frame_length,
                 start=0, stop=None, X_labels=None,
		 _iter_num_batches=None, rng=_default_seed, 
                 load_to_memory=False, cache_size=None,
                 cache_delta=None, schwenk=False):
        """
        Parameters
        ----------
        path : str
            The base path to the data
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
	_iter_num_batches : int, optional
	    Determines number of batches to cycle through in one epoch. Used to
	    calculate the validation score without having to cycle through the
	    entire dataset
        load_to_memory : bool, optional
            If True, will load all requested data into memory. This allows the
            iterations to go faster, but requires significantly more memory.
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
        self.source_path = source_path
        self.target_path = target_path
        assert schwenk == True, "Not implemented for other datasets"

        super(BilingSkipgram, self).__init__(source_path, '', which_set, frame_length,
                 start=start, stop=stop, X_labels=X_labels,
		 _iter_num_batches=_iter_num_batches, rng=rng, 
                 load_to_memory=False, cache_size=cache_size,
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
            source_sequences = [self.source_sequences[i] for i in indexes]
            target_sequences = [self.target_sequences[i] for i in indexes]

            # Get random source word index for "ngram"
            source_i = [numpy.random.randint(self.frame_length/2 +1, len(s)-self.frame_length/2, 1)[0] 
                        for s in source_sequences]
            target_i = [min(abs(int(numpy.random.normal(s_i, self.frame_length/3.0))), len(s)-1)
                        for s_i, s in safe_izip(source_i, target_sequences)]
            

            # Words mapped to integers greater than input max are set to 1 (unknown)
            X = [numpy.asarray([s[i]]) for i, s in safe_izip(source_i, source_sequences)]
            X[X>=self.X_labels] = numpy.asarray([1])
            X = numpy.asarray(X)
            y = [numpy.asarray([s[i]]) for i, s in safe_izip(target_i, target_sequences)]
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
        self.sourceFNs['targets'] = getTarget
        self.sourceFNs['features'] =  getFeatures
        

    def _load_data_helper(self, start, stop):
        with tables.open_file(self.source_path) as f:
            table_name, index_name = '/phrases', '/biling_long_indices'
            indices = f.get_node(index_name)[start:stop]
            words = f.get_node(table_name)
            source_data = [words[i['pos']:i['pos']+i['length']] for i in indices]
        with tables.open_file(self.target_path) as f:
            table_name, index_name = '/phrases', '/biling_long_indices'
            indices = f.get_node(index_name)[start:stop]
            words = f.get_node(table_name)
            target_data = [words[i['pos']:i['pos']+i['length']] for i in indices]
        return (target_data, source_data)

    def _parallel_load_data(self, start, stop, queue):
        queue.put(self._load_data_helper(start, stop))

    def _load_data(self, which_set, startstop):
        """
        Load the data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # TODO: Make files work with this terminology
        start, stop = startstop
        source_data, target_data = self._load_data_helper(start, stop)
        self.source_sequences = source_data
        self.target_sequences = target_data
        self.num_examples = len(source_data)
        return
        print "Got", self.num_examples, "sentences"
        #self.samples_sequences = numpy.asarray(self.samples_sequences)
 
    def _maybe_load_data(self):
       # print "In maybe load data"
        if self._num_since_last_load >= self._cache_delta and not self._loading:
            print "need to load data"

            # If we would go over the end of the dataset by loading more data,
            # we start over from the beginning of the dataset.
            if (self._next_cache_index + self._cache_delta > 
                self._max_data_index):
                start = self._start
                stop = self._cache_delta + start
            else:
                start = self._next_cache_index
                stop = self._cache_delta + start

            self._next_cache_index = stop 
            assert self._loading == False, "Cannot have 2 processes at once"
            self._loading = True
            p = Process(target=self._parallel_load_data, args=(start, stop, self._data_queue))
            p.start()
            #self._parallel_load_data(start, stop, self._data_queue)

        if not self._data_queue.empty():
            #print "queue has stuff"
            new_source, new_target = self._data_queue.get()
            self.source_sequences =  self.source_sequences[self._cache_delta:] + new_source
            self.target_sequences =  self.target_sequences[self._cache_delta:] + new_target
            #print "Queue is empty", self._data_queue.empty()
            self._num_since_last_load = 0
            assert self._data_queue.empty(), "Cannot have 2 things on queue at once"
            self._loading = False
            #print "got stuff from queue"



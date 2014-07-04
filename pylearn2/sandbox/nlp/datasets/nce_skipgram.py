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
from pylearn2.sandbox.nlp.datasets.skipgram import H5Skipgram
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class NoiseDataset(H5Skipgram):
    """
    This a wrapper dataset if you want to use NCE.
    Sampling from multi-nomial is very expeo:
    """

    def __init__(self, noise_p, num_noise, **kwargs):
        super(NoiseDataset, self).__init__(**kwargs)
        self.noise_p = noise_p
        self.num_noise = num_noise
        self._spaces.append(IndexSpace(dim=num_noise, max_labels=10000, dtype='int64'))
        self._source.append('noises')
        space = CompositeSpace(self._spaces)
        self.data_specs = (self._spaces, self._sources)
        
        def getNoise(indexes)

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError('In DenseDesignMatrix.iterator, both the '
                                 '"data_specs" argument and deprecated '
                                 'arguments "topo" or "targets" were '
                                 'provided.',
                                 (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are "
                          "being deprecated, and will be removed "
                          "around November 7th, 2013. `data_specs` "
                          "should be used instead.",
                          stacklevel=2)

            # build data_specs from topo and targets if needed
            if topo is None:
                topo = getattr(self, '_iter_topo', False)
            if topo:
                # self.iterator is called without a data_specs, and with
                # "topo=True", so we use the default topological space
                # stored in self.X_topo_space
                assert self.X_topo_space is not None
                X_space = self.X_topo_space
            else:
                X_space = self.X_space

            if targets is None:
                targets = getattr(self, '_iter_targets', False)
            if targets:
                assert self.y is not None
                y_space = self.data_specs[0].components[1]
                space = CompositeSpace((X_space, y_space))
                source = ('features', 'targets')
            else:
                space = X_space
                source = 'features'

            data_specs = (space, source)
            convert = None

        else:
            if data_specs is None:
                data_specs = self._iter_data_specs

            # If there is a view_converter, we have to use it to convert
            # the stored data for "features" into one that the iterator
            # can return.
            space, source = data_specs
            if isinstance(space, CompositeSpace):
                sub_spaces = space.components
                sub_sources = source
            else:
                sub_spaces = (space,)
                sub_sources = (source,)

            convert = []
            for sp, src in safe_zip(sub_spaces, sub_sources):
                if src == 'features' and \
                   getattr(self, 'view_converter', None) is not None:
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(batch,
                                                                       space))
                else:
                    conv_fn = None

                convert.append(conv_fn)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return NoiseIterator(self,
                             mode(self.X.shape[0],
                                  batch_size,
                                  num_batches,
                                  rng),
                             data_specs=data_specs,
                             return_tuple=return_tuple,
                             convert=convert,
                             noise_p=self.noise_p,
                             num_noise=self.num_noise)



if __name__ == "__main__":
    from pylearn2.sandbox.nlp.datasets.penntree import PennTreebank
    ds = PennTreebank('valid', context_len = 2)
    nds = NoiseDataset(ds, num_noise=5, noise_p = np.load('penntree_unigram.npy'))
    spaces = CompositeSpace(components=(IndexSpace(dim=2, max_labels=10000, dtype='int64'),
                            IndexSpace(dim=1, max_labels=10000, dtype='int64'),
                            IndexSpace(dim=1, max_labels=10000, dtype='int64')))
    sources = ('features', 'targets', 'noises')
    it = nds.iterator(batch_size = 10, mode = 'sequential', targets=True) #, data_specs=(spaces, sources))

    count = 0
    for item in it:
        print item[2]
        count += 1
        if count > 3:
            break

    # test one_billion
    if 0:
        from noisylearn.projects.gating2.one_billion import OneBillionWord
        train = OneBillionWord('train',5)
        iter= train.iterator(mode='sequential',batch_size=100,data_specs=train.data_specs)
        rval = iter.next()
        for i in range(0,100):
            print rval[0][i,:],rval[1][i]
        print 'next'
        rval = iter.next()
        for i in range(0,100):
            print rval[0][i,:],rval[1][i]
        print 'next'
        rval = iter.next()
        for i in range(0,100):
            print rval[0][i,:],rval[1][i]
        print train.num_words
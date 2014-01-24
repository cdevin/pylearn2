import os
import warnings
import functools
import numpy as np
from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets import Dataset
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils import safe_zip
from noisylearn.projects.tiled.iteration import FiniteDatasetIterator

class Wikipedia(dense_design_matrix.DenseDesignMatrix):

    _default_seed = (17, 2, 946)

    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len, char = True, rng = _default_seed):

        if which_set not in self.valid_set_names:
            raise ValueError("which_set should have one of these values: {}".format(self.valid_set_names))
        if char:
            data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "wikipedia-text/{}_chars.npy".format(which_set)))
        else:
            raise NotImplmentedError()

        self.seq_len = seq_len
        self.X = data

        #super(Wikipedia, self).__init__(X = data)

        x_space = VectorSpace(dim = seq_len)
        x_source = 'features'
        y_space = VectorSpace(dim=1)
        y_source = 'targets'

        space = CompositeSpace((x_space, y_space))
        source = (x_source, y_source)
        self.data_specs = (space, source)
        self.X_space = x_space

        self.compress = False
        self.design_loc = None
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = np.random.RandomState(rng)
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_targets = False
        self._iter_data_specs = (self.X_space, 'features')

    def get_data(self):
        return self.X, np.zeros(1, dtype='int32')

    @property
    def num_examples(self):
        return (self.X.shape[0] - self.seq_len)


    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 topo=None, targets=None, rng=None, data_specs=None,
                 return_tuple=False):
        """
        .. todo::

            WRITEME
        """

        if topo is not None or targets is not None:
            if data_specs is not None:
                raise ValueError("In DenseDesignMatrix.iterator, both "
                        "the `data_specs` argument and deprecated arguments "
                        "`topo` or `targets` were provided.",
                        (data_specs, topo, targets))

            warnings.warn("Usage of `topo` and `target` arguments are being "
                    "deprecated, and will be removed around November 7th, "
                    "2013. `data_specs` should be used instead.",
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
                if (src == 'features' and
                        getattr(self, 'view_converter', None) is not None):
                    conv_fn = (lambda batch, self=self, space=sp:
                               self.view_converter.get_formatted_batch(
                                   batch,
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
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                     num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)


def convert_to_pkl():

    # This convert npz file to separate pkl for each set
    # Do this only once

    data = np.load(preprocess("${PYLEARN2_DATA_PATH}/wikipedia-text/enwiki_char_and_word.npz"))
    train = data['train_chars']
    test = data['test_chars']
    valid = data['valid_chars']

    serial.save("${PYLEARN2_DATA_PATH}/wikipedia-text/train_chars.npy", train)
    serial.save("${PYLEARN2_DATA_PATH}/wikipedia-text/valid_chars.npy", valid)
    serial.save("${PYLEARN2_DATA_PATH}/wikipedia-text/test_chars.npy", test)

if __name__ == "__main__":

    #convert_to_pkl()
    train = Wikipedia('train', 100)
    valid = Wikipedia('valid', 100)
    test = Wikipedia('test', 100)
    print train.num_examples
    print valid.num_examples
    print test.num_examples


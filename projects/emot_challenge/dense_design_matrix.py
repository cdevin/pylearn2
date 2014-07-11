"""TODO: module-level docstring."""
__authors__ = "Ian Goodfellow and Mehdi Mirza"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"
import functools

import warnings
import numpy as np
from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    FiniteDatasetIteratorPyTables,
    resolve_iterator_class
)
N = np
import copy
# Don't import tables initially, since it might not be available
# everywhere.
tables = None


from pylearn2.datasets.dataset import Dataset
from pylearn2.datasets.dataset.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets import control
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
from theano import config
from noisy_encoder.scripts.emot_challenge.space import Conv3DSpace



class DenseDesignMatrix(DenseDesignMatrix):
    """
    A class for representing datasets that can be stored as a dense design
    matrix, such as MNIST or CIFAR10.
    """
    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None,
                 view_converter=None, axes = ('b', 0, 1, 'c'),
                 rng=_default_seed, preprocessor = None, fit_preprocessor=False):
        """
        Parameters
        ----------

        X : ndarray, 2-dimensional, optional
            Should be supplied if `topo_view` is not. A design
            matrix of shape (number examples, number features)
            that defines the dataset.
        topo_view : ndarray, optional
            Should be supplied if X is not.  An array whose first
            dimension is of length number examples. The remaining
            dimensions are xamples with topological significance,
            e.g. for images the remaining axes are rows, columns,
            and channels.
        y : ndarray, 1-dimensional(?), optional
            Labels or targets for each example. The semantics here
            are not quite nailed down for this yet.
        view_converter : object, optional
            An object for converting between design matrices and
            topological views. Currently DefaultViewConverter is
            the only type available but later we may want to add
            one that uses the retina encoding that the U of T group
            uses.
        rng : object, optional
            A random number generator used for picking random
            indices into the design matrix when choosing minibatches.
        """
        self.X = X
        self.y = y

        if topo_view is not None:
            assert view_converter is None
            self.set_topological_view(topo_view, axes)
        else:
            assert X is not None, ("DenseDesignMatrix needs to be provided "
                    "with either topo_view, or X")
            if view_converter is not None:
                self.view_converter = view_converter

                # Build a Conv2DSpace from the view_converter
                if not (isinstance(view_converter, DefaultViewConverter)
                        and len(view_converter.shape) == 3):
                    raise NotImplementedError("Not able to build a Conv2DSpace "
                            "corresponding to this converter: %s"
                            % view_converter)

                axes = view_converter.axes
                rows, cols, channels = view_converter.shape

                # self.X_topo_space stores a "default" topological space that
                # will be used only when self.iterator is called without a
                # data_specs, and with "topo=True", which is deprecated.
                self.X_topo_space = Conv2DSpace(
                        shape=(rows, cols), num_channels=channels, axes=axes)
            else:
                self.X_topo_space = None

            # Update data specs, if not done in set_topological_view
            X_space = VectorSpace(dim=self.X.shape[1])
            X_source = 'features'
            if y is None:
                space = X_space
                source = X_source
            else:
                y_space = VectorSpace(dim=self.y.shape[-1])
                y_source = 'targets'

                space = CompositeSpace((X_space, y_space))
                source = (X_source, y_source)
            self.data_specs = (space, source)
            self.X_space = X_space

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

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor

    def __setstate__(self, d):

        if d['design_loc'] is not None:
            if control.get_load_data():
                d['X'] = N.load(d['design_loc'])
            else:
                d['X'] = None

        if d['compress']:
            X = d['X']
            mx = d['compress_max']
            mn = d['compress_min']
            del d['compress_max']
            del d['compress_min']
            d['X'] = 0
            self.__dict__.update(d)
            if X is not None:
                self.X = N.cast['float32'](X) * mx / 255. + mn
            else:
                self.X = None
        else:
            self.__dict__.update(d)

        # To be able to unpickle older data after the addition of
        # the data_specs mechanism
        if not all(m in d for m in ('data_specs', 'X_space',
                                    '_iter_data_specs', 'X_topo_space')):
            X_space = VectorSpace(dim=self.X.shape[1])
            X_source = 'features'
            if self.y is None:
                space = X_space
                source = X_source
            else:
                y_space = VectorSpace(dim=self.y.shape[-1])
                y_source = 'targets'

                space = CompositeSpace((X_space, y_space))
                source = (X_source, y_source)

            self.data_specs = (space, source)
            self.X_space = X_space
            self._iter_data_specs = (X_space, X_source)

            view_converter = d.get('view_converter', None)
            if view_converter is not None:
                # Build a Conv2DSpace from the view_converter
                if not (isinstance(view_converter, DefaultViewConverter)
                        and len(view_converter.shape) == 3):
                    raise NotImplementedError(
                            "Not able to build a Conv2DSpace "
                            "corresponding to this converter: %s"
                            % view_converter)

                axes = view_converter.axes
                rows, cols, channels = view_converter.shape

                # self.X_topo_space stores a "default" topological space that
                # will be used only when self.iterator is called without a
                # data_specs, and with "topo=True", which is deprecated.
                self.X_topo_space = Conv2DSpace(
                        shape=(rows, cols), num_channels=channels, axes=axes)

    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of training
            examples.
        TODO: why is this parameter named 'V'?
        """
        assert not N.any(N.isnan(V))
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        sequence_length = V.shape[axes.index('t')]
        self.view_converter = DefaultViewConverter([rows, cols, channels], axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = Conv3DSpace(
                shape=(rows, cols), num_channels=channels, sequence_length = sequence_length, axes=axes)
        assert not N.any(N.isnan(self.X))

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            y_space = VectorSpace(dim=self.y.shape[-1])
            y_source = 'targets'
            space = CompositeSpace((X_space, y_space))
            source = (X_source, y_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

    def set_view_converter_axes(self, axes):
        """
        Change the axes of the view_converter, if any.

        This function is only useful if you intend to call self.iterator
        without data_specs, and with "topo=True", which is deprecated.
        """
        assert self.view_converter is not None
        warnings.warn("Rather than setting the axes of a dataset's "
                "view_converter, then building an iterator with "
                "'topo=True', which is deprecated, you can simply "
                "build an iterator with "
                "'data_specs=Conv2DSpace(..., axes=axes)'.",
                stacklevel=3)

        self.view_converter.axes = axes
        # Update self.X_topo_space, which stores the "default"
        # topological space
        rows, cols, channels = self.view_converter.shape
        self.X_topo_space = Conv2DSpace(
                shape=(rows, cols), num_channels=channels, axes=axes)



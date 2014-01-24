import numpy as np
from pylearn2.utils.iteration import SubsetIterator
from pylearn2.utils.iteration import FiniteDatasetIterator as FiniteDatasetIteratorBase

class SequentialSubsetIterator(SubsetIterator):
    """
    .. todo::

        WRITEME
    """
    def __init__(self, dataset_size, batch_size, num_batches, rng=None):
        """
        .. todo::

            WRITEME
        """
        if rng is not None:
            raise ValueError("non-None rng argument not supported for "
                             "sequential batch iteration")
        assert num_batches is None or num_batches >= 0
        self._dataset_size = dataset_size
        if batch_size is None:
            if num_batches is not None:
                batch_size = int(np.ceil(self._dataset_size / num_batches))
            else:
                raise ValueError("need one of batch_size, num_batches "
                                 "for sequential batch iteration")
        elif batch_size is not None:
            if num_batches is not None:
                max_num_batches = np.ceil(self._dataset_size / batch_size)
                if num_batches > max_num_batches:
                    raise ValueError("dataset of %d examples can only provide "
                                     "%d batches with batch_size %d, but %d "
                                     "batches were requested" %
                                     (self._dataset_size, max_num_batches,
                                      batch_size, num_batches))
            else:
                num_batches = np.ceil(self._dataset_size / batch_size)
        self._batch_size = batch_size
        self._num_batches = num_batches
        self._next_batch_no = 0
        self._idx = 0
        self._batch = 0

    def next(self):
        """
        .. todo::

            WRITEME
        """
        if self._batch >= self.num_batches or self._idx >= self._dataset_size:
            raise StopIteration()

        # this fix the problem where dataset_size % batch_size != 0
        elif (self._idx + self._batch_size) > self._dataset_size:
            self._last = slice(self._idx, self._dataset_size)
            self._idx = self._dataset_size
            return self._last

        else:
            self._last = slice(self._idx, self._idx + self._batch_size)
            self._idx += self._batch_size
            self._batch += 1
            return self._last

    fancy = False
    stochastic = False

    @property
    def num_examples(self):
        """
        .. todo::

            WRITEME
        """
        product = self.batch_size * self.num_batches
        return min(product, self._dataset_size)

    @property
    def uneven(self):
        """
        .. todo::

            WRITEME
        """
        return self.batch_size * self.num_batches > self._dataset_size



class FiniteDatasetIterator(FiniteDatasetIteratorBase):
    def next(self):

        next_index = self._subset_iterator.next()
        ind = self._raw_data[0][next_index]
        batch_size = len(ind)

        targets = False
        if len(self._raw_data) == 2:
            targets = True
            y = np.zeros((batch_size, 1))
        x = np.zeros((batch_size, self._dataset.seq_len))

        for i in xrange(batch_size):
            x[i] = self._raw_data[0][ind[i]:ind[i] + self._dataset.seq_len]
            if targets:
                y[i] = self._raw_data[0][ind[i] + self._dataset.seq_len]

        if targets:
            rval = (self._convert[0](x), self._convert[1](y))
        else:
            rval = (self._convert[0](x))

        if not self._return_tuple and len(rval) == 1:
            rval, = rval
        return rval



import numpy as np
from pylearn2.utils.iteration import FiniteDatasetIterator as FiniteDatasetIteratorBase


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



import numpy as np
from pylearn2.utils.iteration import SubsetIterator
from pylearn2.utils.iteration import FiniteDatasetIterator


class SequenceDatasetIterator(FiniteDatasetIterator):

    def get_seq(self, ind):
        """
        return seq_len words before ind, including ind
        while paying attention to sentence structure and filling
        beginning and end of setences with special character
        """

        return format_sentence(data = self._raw_data[0],
                        seq_len = self._dataset.seq_len,
                        ind = ind - 1,
                        begin = self._dataset.begin_sentence,
                        end = self._dataset.end_sentence)

    def next(self):

        next_index = self._subset_iterator.next()
        targets = False
        aux_targets = False

        y = self._raw_data[0][next_index].reshape((self.batch_size, 1))

        if isinstance(next_index, slice):
            next_index = slice_to_list(next_index)

        x = np.zeros((self.batch_size, self._dataset.seq_len))
        x = np.asarray([self.get_seq(i) for i in xrange(self.batch_size)])

        y = self._dataset.mapped_dict[y]

        rval = (self._convert[0](x), self._convert[1](y))

        return rval


def slice_to_list(item):
    ifnone = lambda a, b: b if a is None else a
    return list(range(ifnone(item.start, 0), item.stop, ifnone(item.step, 1)))


def format_sentence(data, ind, seq_len, begin, end):
    """

    Parameters
    ----------
    begin: int
        index of the start of sentence <S>
    end: index of end of sentence </S>
    """

    rval = np.ones((seq_len)) * end
    if ind > seq_len:
        rval[:] = data[ind-seq_len:ind].flatten()
    elif ind > 0:
        rval[seq_len-ind:] =  data[:ind].flatten()

    w = np.where(rval == -1)[0]
    if len(w) > 0:
        rval[0:max(0, w[-1])] = end
        rval[w[-1]] = begin

    return rval



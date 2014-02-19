import os
import numpy as np
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.iteration import resolve_iterator_class
from noisylearn.projects.tiled.dataset import SequenceDesignMatrix

class OneBilllionWords(SequenceDesignMatrix):

    _default_seed = (17, 2, 946)

    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len, rng=_default_seed):

        if which_set not in self.valid_set_names:
            raise ValueError("which_set should have one of these values: {}".format(self.valid_set_names))
        data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                "smt/billion/en/newsxx.test.npy")

        self.seq_len = seq_len
        self.X = data

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


if __name__ == "__main__":

    train = OneBilllionWords('train', 6)
    print train.num_examples


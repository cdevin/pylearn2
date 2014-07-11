import os
import numpy as np
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.space import VectorSpace, CompositeSpace
from pylearn2.utils.iteration import resolve_iterator_class
from noisylearn.projects.tiled.dataset import SequenceClusterDesignMatrix
from noisylearn.utils.cache import CachedAttribute
from noisylearn.projects.tiled.flatten_one_billion_word import get_num_words
from pylearn2.utils.string_utils import preprocess
from noisylearn.projects.tiled.brown_utils import map_words, BrownClusterDict

class OneBilllionWords(SequenceClusterDesignMatrix):

    _default_seed = (17, 2, 946)

    valid_set_names = ["train","valid", "test"]
    def __init__(self, which_set, seq_len, brown = None, rng=_default_seed):

        if which_set not in self.valid_set_names:
            raise ValueError("which_set should have one of these values: {}".format(self.valid_set_names))
        #data = serial.load(os.path.join("${PYLEARN2_DATA_PATH}",
                #"smt/billion/en/newsxx.test.npy"))
        if which_set == 'train':
            data = np.memmap(preprocess(os.path.join("${PYLEARN2_DATA_CUSTOM}", "obw/{}.dat".format(which_set))),
                             mode = 'r')
        else:
            data = serial.load(os.path.join("${PYLEARN2_DATA_CUSTOM}", "obw/{}.npy".format(which_set)))

        if data.ndim == 1:
            data = data.reshape((data.shape[0], 1))

        self.seq_len = seq_len
        self.X = data
        self.y = None
        self.brown = brown

        x_space = VectorSpace(dim = seq_len)
        y_space = VectorSpace(dim=1)
        cls_space = VectorSpace(dim=brown)
        space = CompositeSpace((x_space, y_space, cls_space))
        source = ('features', 'targets', 'aux_targets')

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


        assert brown is not None
        cluster_path = os.path.join("${PYLEARN2_DATA_CUSTOM}", "obw/brown/{}.dat".format(brown))
        cluster_path = preprocess(cluster_path)
        clusters = BrownClusterDict(preprocess(cluster_path))
        word_dict = serial.load("${PYLEARN2_DATA_PATH}/smt/billion/en/newsxx_word_indxs.pkl")
        # add extra chars
        word_dict['<s>'] = self.begin_sentence
        word_dict['</s>'] = self.end_sentence
        mapped_dict, clusters, _ = map_words(word_dict, clusters)
        self.clusters = clusters
        self.mapped_dict = mapped_dict


    @CachedAttribute
    def num_words(self):
        rval = get_num_words()
        return rval

    @property
    def end_sentence(self):
        return self.num_words + 1

    @property
    def begin_sentence(self):
        return self.num_words + 2


if __name__ == "__main__":

    train = OneBilllionWords('test', 6, brown = 100)
    #mapping = DataSpecsMapping(train.data_specs)
    #flattened_specs = (mapping.flatten(train.data_specs[0], return_tuple=False), mapping.flatten(train.data_specs[1]))
    iter = train.iterator(mode = 'sequential', batch_size = 100, data_specs = train.data_specs)
    rval = iter.next()
    #import ipdb
    #ipdb.set_trace()
    print train.num_words


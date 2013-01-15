import numpy
from pylearn2.datasets import dense_design_matrix
from pylearn2.utils.serial import load

class Lisa(dense_design_matrix.DenseDesignMatrix):

    mapper = {'valid' :0, 'test': 1}

    def __init__(self, which_set, one_hot = False,
                 image_size = 48, example_range = None,
                 center = False, shuffle=False, rng=None, seed=132987):

        assert which_set in self.mapper.keys()

        path = '${PYLEARN2_DATA_PATH}/faces/lisa/lisa.pkl'
        path = '/data/lisatmp2/mirzamom/data/lisa_preprocessed.pkl'
        data = load(path)

        data_x = data['image']
        data_y = data['emotion']
        data_y_idendity = data['idendity']

        if which_set == 'test':
            # the first 978 images are images from idendities 0..16
            data_x = data_x[:979]
            data_y = data_y[:979]
            data_y_idendity = data_y_idendity[:979]
        else:
            data_x = data_x[979:]
            data_y = data_y[979:]
            data_y_idendity = data_y_idendity[979:]

        if center:
            data_x -= 127.5

        if shuffle:
            rng = rng if rng else numpy.random.RandomState(seed)
            rand_idx = rng.permutation(len(data_x))
            data_x = data_x[rand_idx]
            data_y = data_y[rand_idx]
            data_y_idendity = data_y_idendity[rand_idx]

        if one_hot:
            one_hot = numpy.zeros((data_y.shape[0], 7), dtype='float32')
            for i in xrange(data_y.shape[0]):
                one_hot[i, data_y[i]] = 1.
            data_y = one_hot

        # create view converting for retrieving topological view
        view_converter = dense_design_matrix.DefaultViewConverter((image_size, image_size, 1))

        # init the super class
        super(Lisa, self).__init__(X = data_x, y = data_y, view_converter = view_converter)

        self.y_idendity = data_y_idendity

        assert not numpy.any(numpy.isnan(self.X))

if __name__ == "__main__":
    Lisa("valid")

from pylearn2.datasets.tfd import TFD
from pylearn2.space import VectorSpace, CompositeSpace

class MyTFD(TFD):

    def __init__(self, which_set, fold = 0, image_size = 48,
                 example_range = None, center = False, scale = False,
                 shuffle=False, one_hot = False, rng=None, seed=132987,
                 preprocessor = None, axes = ('b', 0, 1, 'c')):
        super(MyTFD, self).__init__(which_set, fold, image_size, example_range,
                center, scale, shuffle, one_hot, rng, seed, preprocessor, axes)
        specs, source = self.data_specs
        specs = [specs.components[0]] + [VectorSpace(dim=7), VectorSpace(dim=7)]
        source = list(source) + ['second_targets']
        self.data_specs = (CompositeSpace(specs), tuple(source))

    def get_data(self):
        return self.X, self.y, self.y

if __name__ == "__main__":
    an = MyTFD('valid')

import argparse
from pylearn2.utils import serial
from pylearn2.datasets import preprocessing
from pylearn2.datasets.cifar10 import CIFAR10
from utils.config import get_data_path
from DLN.datasets.preprocessing import Scale
from sklearn.decomposition import PCA as sk_pca

class PCA():
    """ PyLearn2 PCA is broken, so doing it manually for time being"""

    def __init__(self, num_comp):
        self.num_comp = num_comp

    def train(self, data):
        self.pca = sk_pca(self.num_comp)
        self.pca.fit(data)

    def __call__(self, data):
        return self.pca.transform(data)

def make(method):

    if method == "zca":
        print 'Loading CIFAR_10 train set...'
        train = CIFAR10(which_set="train", rescale = True, center = True)

        print "Preparing output directory..."
        DATA_PATH = get_data_path()
        data_dir = DATA_PATH + "cifar10_local"
        output_dir = data_dir + '/pylearn2'
        serial.mkdir( output_dir )


        pipeline = preprocessing.Pipeline()
        pipeline.items.append(preprocessing.GlobalContrastNormalization())
        pipeline.items.append(preprocessing.ZCA())
        train.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        train.use_design_loc(output_dir + '/train.npy')
        serial.save(output_dir + '/train.pkl', train)

        # Test
        test = CIFAR10(which_set="test", rescale = True, center = True)
        test.apply_preprocessor(preprocessor=pipeline, can_fit=True)
        test.use_design_loc(output_dir + '/test.npy')
        serial.save(output_dir + '/test.pkl', test)

        serial.save(output_dir + '/preprocessor.pkl',pipeline)

    elif method == "pca":
        print 'Loading CIFAR_10 train set...'
        train = CIFAR10(which_set="train", rescale = True, center = True)

        print "Preparing output directory..."
        DATA_PATH = get_data_path()
        data_dir = DATA_PATH + "cifar10_local"
        output_dir = data_dir + '/pylearn2_pca'
        serial.mkdir( output_dir )


        pipeline = preprocessing.Pipeline()
        #pipeline.items.append(preprocessing.GlobalContrastNormalization())
        #pipeline.items.append(preprocessing.ZCA())
        pipeline.items.append(preprocessing.MakeUnitNorm())
        train.apply_preprocessor(preprocessor=pipeline, can_fit=True)

        pca = PCA(512)
        pca.train(train.get_design_matrix())
        train.set_design_matrix(pca(train.get_design_matrix()))

        train.use_design_loc(output_dir + '/train.npy')
        serial.save(output_dir + '/train.pkl', train)
        # Test
        test = CIFAR10(which_set="test", rescale = True, center = True)
        test.apply_preprocessor(preprocessor=pipeline, can_fit=True)

        test.set_design_matrix(pca(test.get_design_matrix()))

        test.use_design_loc(output_dir + '/test.npy')
        serial.save(output_dir + '/test.pkl', test)

        serial.save(output_dir + '/preprocessor.pkl',pipeline)

    else:
        raise NameError('Wron method option: {}'.format(method))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'CIFAR10 dataset maker')
    parser.add_argument('-m', '--method', choices = ['pca', 'zca'], required = True)
    args = parser.parse_args()
    make(args.method)

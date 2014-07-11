import argparse
from noisy_encoder.utils.io import load_data
from pylearn2.utils import serial
from utils.config import get_data_path, get_result_path
from noisy_encoder.training_algorithms.eval import eval


RESULT_PATH = get_result_path()
DATA_PATH = get_data_path()

def evaluate(dataset, data_path, fold, model, batch_size):

    datasets = load_data(dataset,
                        data_path,
                        False,
                        False,
                        False,
                        fold)
    model = serial.load(model)
    test_score, valid_score = eval(model, datasets, batch_size)

    print "Test score: {},\tValid score: {}".format(test_score, valid_score)

    return test_score, valid_score




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'supervised trainer')
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10',
        'cifar100', 'timit', 'tfd', 'tfd_new_conv', 'siamese'], required = True)
    parser.add_argument('-m', '--model')
    parser.add_argument('-p', '--path')
    args = parser.parse_args()

    batch_size = 100
    evaluate(args.dataset, args.path, 0, args.model, batch_size)

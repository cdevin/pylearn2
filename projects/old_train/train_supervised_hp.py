import os
import numpy
import argparse
from jobman.tools import DD
from utils.config import get_data_path, get_result_path
from noisy_encoder.utils.io import load_data
from noisy_encoder.training_algorithms.sgd import sgd, sgd_mix, sgd_large
from noisy_encoder.models.conv import LeNetLearner, LeNetLearnerMultiCategory
from noisy_encoder.models.siamese import Siamese, SiameseVariant, SiameseMix, SiameseMixSingleCategory
from noisy_encoder.utils.corruptions import BinomialCorruptorScaled
from theano.tensor.shared_randomstreams import RandomStreams
from hyperopt import hp

RESULT_PATH = get_result_path()
DATA_PATH = get_data_path()

def load_model(space, numpy_rng, theano_rng):
    if space["model"] == 'conv':
        return LeNetLearner(
                conv_layers = space["conv_layers"],
                batch_size = space["batch_size"],
                mlp_act = space["mlp_act"],
                mlp_input_corruption_levels = space["mlp_input_corruption_levels"],
                mlp_hidden_corruption_levels = space["mlp_hidden_corruption_levels"],
                mlp_nunits = space["mlp_nunits"],
                n_outs = space["n_outs"],
                irange = space["irange"],
                bias_init = space["bias_init"],
                random_filters = space["random_filters"],
                rng = numpy_rng)
    else:
        raise NameError("Unknown model: {}".format(space["model"]))

def experiment(space):

    numpy_rng = numpy.random.RandomState(89677)
    theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

    datasets = load_data(space["dataset"],
                        space["data_path"],
                        space["shuffle"],
                        space["scale"],
                        space["norm"],
                        space["fold"])
    model = load_model(space, numpy_rng, theano_rng)
    if space["train_alg"] == "sgd":
        train_alg = sgd
    else:
        raise NameError("Unknown training algorithms: {}".format(train_alg))
    test_score, valid_score = train_alg(model = model,
                                datasets = datasets,
                                training_epochs = space["nepochs"],
                                batch_size = space["batch_size"],
                                coeffs = space["coeffs"],
                                lr_params = space["lr_params"],
                                save_frequency = space["save_frequency"],
                                save_name = space["save_name"],
                                enable_momentum = space["enable_momentum"],
                                momentum_params = space["momentum_params"])

    from hyperopt import STATUS_OK
    return {'loss' : test_score, 'status': STATUS_OK}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'supervised trainer')
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10',
        'cifar100', 'timit', 'tfd', 'tfd_conv', 'siamese', 'siamese_variant',
        'conv_google', 'siamese_google', 'tfd_siamese_mix', 'google_large_conv'], required = True)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        mnist_experiment()
    elif args.dataset == 'cifar10':
        cifar10_experiment()
    elif args.dataset == 'cifar100':
        cifar100_experiment()
    elif args.dataset == 'tfd':
        tfd_experiment()
    elif args.dataset == 'mnsit':
        mnist_experiment()
    elif args.dataset == 'siamese':
        siamese_experiment()
    elif args.dataset == 'siamese_variant':
        siamese_variant_experiment()
    elif args.dataset == 'tfd_conv':
        tfd_conv_experiment()
    elif args.dataset == 'conv_google':
        google_conv_experiment()
    elif args.dataset == 'siamese_google':
        google_siamese_experiment()
    elif args.dataset == 'tfd_siamese_mix':
        tfd_siamese_mix_experiment()
    elif args.dataset == 'google_large_conv':
        google_large_conv_experiment()

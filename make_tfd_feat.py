import argparse, pickle, os
import numpy
from io import load_tfd, features


def get_features(model, data_path, ds_type, fold):
    data_x, data_y = load_tfd(data_path, fold, ds_type)
    data_x = features(model, data_x)

    return data_x, data_y


def save(data, path):

    with open(path, 'wb') as outf:
        pickle.dump(data, outf)

def main(model, input, out, folds):

    #unlabeld
    print "Unlabled set"
    try:
        os.mkdir(out + "unsupervised")
    except OSError:
        pass

    fname = out + "unsupervised/TFD_unsupervised_train_unlabeled_all.pkl"
    save(get_features(model, input, 'unlabled', -1), fname)

    for fold in folds:
        print "Fold{}".format(fold)
        try:
            os.mkdir(out + "FOLD{}".format(fold))
        except OSError:
            pass

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "train_labeled")
        save(get_features(model, input, 'train', fold), fname)

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "valid")
        save(get_features(model, input, 'valid', fold), fname)

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "test")
        save(get_features(model, input, 'test', fold), fname)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Make a TFD dataset with "\
            "same format as oirginal data with features from the input model")
    parser.add_argument('-m', '--model', required = True,
            help = "Model file")
    parser.add_argument('-o', '--out', required = True,
            help = "Output path")
    parser.add_argument('-i', '--input', required = True)

    args = parser.parse_args()

    main(args.model, args.input, args.out, [0])



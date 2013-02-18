import argparse, pickle, os
import numpy
from util.funcs import load_tfd, features


def get_features(model, data_path, ds_type, fold, scale):
    data_x, data_y = load_tfd(data_path = data_path,
                            fold = fold,
                            ds_type = ds_type,
                            scale = scale)
    data_x = features(model, data_x)

    return data_x, data_y


def save(data, path):

    with open(path, 'wb') as outf:
        pickle.dump(data, outf)

def main(model, input, out, folds, scale):

    #unlabeld
    print "Unlabled set"
    try:
        os.mkdir(out + "unsupervised")
    except OSError:
        pass

    fname = out + "unsupervised/TFD_unsupervised_train_unlabeled_all.pkl"
    save(get_features(model, input, 'unlabled', -1, scale), fname)

    for fold in folds:
        print "Fold{}".format(fold)
        try:
            os.mkdir(out + "FOLD{}".format(fold))
        except OSError:
            pass

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "train_labeled")
        save(get_features(model, input, 'train', fold, scale), fname)

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "valid")
        save(get_features(model, input, 'valid', fold, scale), fname)

        fname = out + "FOLD{}/TFD_RAW_FOLD_{}_{}0.pkl".format(fold, fold, "test")
        save(get_features(model, input, 'test', fold, scale), fname)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Make a TFD dataset with "\
            "same format as oirginal data with features from the input model")
    parser.add_argument('-m', '--model', required = True,
            help = "Model file")
    parser.add_argument('-o', '--out', required = True,
            help = "Output path")
    parser.add_argument('-i', '--input', required = True)
    parser.add_argument('-s', '--scale', default = False, action = "store_true")

    args = parser.parse_args()

    main(args.model, args.input, args.out, [0], args.scale)



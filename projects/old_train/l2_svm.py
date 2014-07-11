import subprocess, os
import argparse
from jobman.tools import DD
from utils.config import get_tmp_path
from noisy_encoder.scripts.train.make_features import convert, load_data, save

def runProcess(exe):
    p = subprocess.Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    while(True):
      retcode = p.poll() #returns None while subprocess is running
      line = p.stdout.readline()
      yield line
      if(retcode is not None):
        break

def str_wrap(ls):
    """Convert list to matlab readable str"""
    res = "["
    for item in ls:
        res += str(item) + " "
    res.rstip(' ')
    return res + "]"

def classify(data, labels, C_vals, standardize, num_train):
    #for line in runProcess(['./l2_svm', '~/Desktop/mnist_l1_299.mat', '~/Desktop/labels.mat', '[1 10]', 'False', '50000']):
    for line in runProcess(['/RQusagers/mirzameh/projects/noisy_encoder/scripts/train/l2_svm', data, labels, C_vals, str(standardize), str(num_train)]):
        print line.rstrip('\n')
        if line.find('Best_C') != -1:
            best_c = float(line.split()[-1])
        if line.find('Train_accuracy') != -1:
            train_acc = float(line.split()[-1].rstrip('%'))
        if line.find('Test_accuracy') != -1:
            test_acc = float(line.split()[-1].rstrip('%'))
        if line.find('Valid_accuracy') != -1:
            valid_acc = float(line.split()[-1].rstrip('%'))

    return best_c, train_acc, valid_acc, test_acc

def get_features(dataset, model_path):

    train_set, test_set = load_data(args.dataset)
    train_feat, test_feat = convert(train_set.X, test_set.X, model_path, dataset, False, False)
    save_path = get_tmp_path()
    name = model_path.split('/')[-1].rstrip('.pkl')
    save(train_feat, test_feat, save_path, name, 'mat')
    save(train_set.y, test_set.y, save_path, 'labels', 'mat')

    return save_path + name, save_path + 'labels.mat'

def experiment (state, channel):

    if state.dataset == 'mnist':
        state.num_train = 50000
    elif state.dataset in ['cifar10', 'cifar10_bw']:
        state.num_train = 45000
    else:
        raise ValueError('Wrong dataset type: {}'.state.dataset)

    # make features if necessary
    if state.data_path[-3:] == 'pkl':
        data_path, labels_path = get_features(state.dataset, state.data_path)
    else:
        data_path = state.data_path
        labels_path = state.labels_path

    state.best_c, state.train_acc, state.valid_acc, state.test_acc = classify(
            data_path, labels_path, state.c_vals, state.standardize, state.num_train)

    # clean up feature files
    if state.data_path[-3:] == 'pkl':
        try:
            os.remove(data_path)
            os.remove(labels_path)
        except OSError:
            pass

    return channel.COMPLETE



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = 'save features of model')
    parser.add_argument('-p', '--datapath', help = "path to data", required=True)
    parser.add_argument('-l', '--label', help = "path to labels")
    parser.add_argument('-s', '--standard', action = "store_true", default = False, help = "standardize data")
    parser.add_argument('-d', '--dataset', choices = ['mnist', 'cifar10', 'cifar10_bw'], required = True)
    parser.add_argument('-c', '--cvals', help = "C values")
    args = parser.parse_args()

    state = DD()
    state.dataset = args.dataset
    state.data_path = args.datapath
    state.labels_path = args.label
    state.standardize = args.standard
    state.c_vals = args.cvals

    experiment(state, None)

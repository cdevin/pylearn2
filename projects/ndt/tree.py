from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
from theano import tensor as T
import numpy as np
import subprocess


ROOT_YAML = 'exp/root.yaml'
INDEX_PATH = 'exp/'


def load_model(model_path):
    model = serial.load(model_path)

    ds = model.dataset_yaml_src
    ds = yaml_parse.load(ds)

    return model, ds

def branch_funbc(model):
    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    y = T.gt(y, 0.5)
    return function([X], y)

def branch_data(brancher, data, batch_size = 100):

    res = []
    for i in xrange(data.shape[0] / batch_size):
        res.append(brancher(data[i * batch_size : (i+1) * batch_size, :]))
    rem = np.mod(data.shape[0], batch_size)
    if rem != 0:
        res.append(brancher(data[data.shape[0] - rem, :]))


    res = np.concatenate(res)
    pos = np.nonzero(res)
    neg = np.where(res == 0)
    return pos[0], neg[0], res

def get_branches(node_id):

    model_path = "{}/{}_{}.pkl".format(MODEL_PATH, MODEL_PREFIX, node_id)
    model, ds = load_model(model_path)
    brancher = branch_funbc(model)
    right, left = branch_data(brancher, ds.X)
    right_name = "{}-R".format(node_id)
    left_name = "{}-L".format(node_id)
    numpy.save("{}/indexes_{}.npy".format(INDEX_PATH, right_name), right)
    numpy.save("{}/indexes_{}.npy".format(INDEX_PATH, left_name), left)
    return right_name, left_name

def read_indexes(node_id):

    right_id = node_id * 2 + 1
    left_if = node_id * 2

    right = "{}/indexes_{}.npy".format(INDEX_PATH, right_name)
    left = "{}/indexes_{}.npy".format(INDEX_PATH, left_name)
    return right, left

def get_yaml(node_id, right, left):
    if node_id == '0':
        return ROOT_YAML

def make_tree(node_id):

        if node_id > 20:
            return

        if node_id == 0:
            right = None
            left = None
        else:
            right, left = read_indexes(node_id)

        yaml_file = get_yaml(node_id, right, left)
        subprocess.Popen([sys.executable, 'script.py {}'.format(yaml_file)],
                            creationflags = subprocess.CREATE_NEW_CONSOLE)

        # after it's done:
        right, left = get_branches(node_id)
        make_tree(right)
        make_tree(left)


def tmp_test():

    model_path = 'exp/mnist_sigmoid_single.pkl'
    model, ds = load_model(model_path)
    brancher = branch_funbc(model)
    right, left, res = branch_data(brancher, ds.X)
    #import ipdb
    #ipdb.set_trace()
    print len(right), len(left)
    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":

    tmp_test()
    #make_tree(0)


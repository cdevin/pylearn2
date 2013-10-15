from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from pylearn2.utils.string_utils import preprocess
from theano import function
from theano import tensor as T
import numpy as np
import argparse


"""
You have to run this with id of one of the left or right node id you want
to produce. Then run both manually
"""

ROOT_YAML = 'exp/root.yaml'
CHILD_YAML = 'exp/child.yaml'
YAML_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"
MODEL_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"
INDEX_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"


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
        res.append(brancher(data[data.shape[0] - rem:, :]))

    res = np.concatenate(res)
    pos = np.nonzero(res)
    neg = np.where(res == 0)
    return pos[0], neg[0]

def get_branches(node_id):
    model_path = "{}/{}_best.pkl".format(MODEL_PATH, node_id)
    model, ds = load_model(model_path)
    brancher = branch_funbc(model)
    right, left = branch_data(brancher, ds.X)

    right_name = "{}/{}_indexes.npy".format(INDEX_PATH, node_id * 2 + 1)
    left_name = "{}/{}_indexes.npy".format(INDEX_PATH, node_id * 2)
    serial.save(right_name, right)
    serial.save(left_name, left)
    return right_name, left_name

def get_yaml(node_id, index):
    args = {'save_path' : "{}/{}_best.pkl".format(MODEL_PATH, node_id)}
    args['index'] = index
    if node_id == 1:
        yaml_f = ROOT_YAML
    else:
        yaml_f = CHILD_YAML

    with open(yaml_f, 'r') as f:
        yaml = f.read()
    return yaml % (args)

def write_job(node_id,yaml_file):
    with open(preprocess("{}/{}.yaml".format(YAML_PATH, node_id)), 'w') as f:
        f.write(yaml_file)

def make_tree(node_id):
        if node_id > 500:
            return

        if node_id == 1:
            yaml_file = get_yaml(node_id, None)
            write_job(node_id,yaml_file)
        else:
            parent_id = int(np.floor(node_id/2.))
            right, left = get_branches(parent_id)

            yaml_file = get_yaml(parent_id * 2 + 1, right)
            write_job(parent_id * 2 + 1, yaml_file)
            yaml_file = get_yaml(parent_id * 2, left)
            write_job(parent_id * 2, yaml_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'nntree trainer')
    parser.add_argument('-i', '--id', type = int, required = True, help = 'Node id')
    args = parser.parse_args()

    make_tree(args.id)

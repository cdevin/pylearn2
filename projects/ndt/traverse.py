import os
import subprocess
import argparse
import numpy as np
from pylearn2.utils.string_utils import preprocess
from noisylearn.projects.ndt.utils import LeafNode
from noisylearn.projects.ndt.tree import make_tree

ROOT_YAML = 'exp/root.yaml'
CHILD_YAML = 'exp/child.yaml'
YAML_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"
MODEL_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"
INDEX_PATH = "${PYLEARN2_EXP_RESULTS}/tree/1"
MAX_NODES = 200

RUN_LIST = []

LEAF_LABELS = {}

def check_node(node_id):
    print node_id
    if node_id > MAX_NODES:
        return

    if os.path.isfile(preprocess("{}/{}.yaml".format(YAML_PATH,node_id))):
        if os.path.isfile(preprocess("{}/{}_best.pkl".format(MODEL_PATH, node_id))):
            check_node(node_id * 2)
            check_node(node_id * 2 + 1)
        else:
            print "need to run"
            RUN_LIST.append(node_id)
    else:
        try:
            print "Making node: {}".format(node_id)
            make_tree(node_id)
        except LeafNode as e:
            print e
            return

def traverse():
    check_node(1)
    print "Run list", RUN_LIST
    np.savetxt('runlist.txt', RUN_LIST)

def run_node():
    RUN_LIST = np.loadtxt('runlist.txt').astype(int)
    try:
        RUN_LIST = list(RUN_LIST)
    except TypeError:
        RUN_LIST = [int(RUN_LIST)]
    node_id = RUN_LIST[0]
    # temporarily remove from list
    RUN_LIST.remove(node_id)
    np.savetxt('runlist.txt', RUN_LIST)

    yaml_f = preprocess("{}/{}.yaml".format(YAML_PATH,node_id))
    proc = subprocess.call("python ~/projects/pylearn2/pylearn2/scripts/train.py  {}".format(yaml_f), shell=True)
    if proc != 0:
        RUN_LIST.append(node_id)
        np.savetxt('runlist.txt', RUN_LIST)
        print "Failed {}".format(node_id)

def assign_tree(node_id = 1):

    if node_id * 2 MAX_NODES:
        # it's a leaf
    else:
        try:
            make_tree(node_id)
        except LeafNode as e:
            # it's a leaf0



if __name__ == "__main__":

    parser  = argparse.ArgumentParser(description = 'Tree man!')
    parser.add_argument('-t', '--task', choices = ['traverse', 'run'])
    args = parser.parse_args()

    if args.task == 'traverse':
        traverse()
    elif args.task == 'run':
        run_node()


import numpy
import subprocess
import argparse
import ipdb


PYLEARN_PATH = "/RQusagers/mirzameh/projects/pylearn2/pylearn2/scripts/"
DATA_PATH = "/RQexec/mirzameh/jobs/mirzamom_db/tree_cifar10/"


def run(exp_ind, data_path, criteria = "valid_y_misclass"):
    cmd = "THEANO_FLAGS=device=gpu python {}print_monitor.py {}{}/best.pkl".format(PYLEARN_PATH, data_path, exp_ind)
    try:
        output = subprocess.check_output(cmd, shell = True)
    except subprocess.CalledProcessError:
        print "failed"
        return 100.

    output = output[output.find(criteria):].split('\n')[0].split(' ')[-1]
    return float(output)

def select(items, data_path, criteria):
    vals = []
    for item in items:
        vals.append(run(item, data_path, criteria))

    print "Results folder: {}, value: {}".format(items[numpy.argmin(vals)], numpy.min(vals))


def child_perf():
    print "node 4-"
    select(xrange(1,7))
    print "node 5-"
    select(xrange(7,13))
    print "node 6-"
    select(xrange(13,19))
    print "node 7-"
    select(xrange(19,25))

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description = 'Selects the best job')
    parser.add_argument('-p', '--path')
    parser.add_argument('-c', '--criteria')
    args = parser.parse_args()
    select(xrange(42), args.path, args.criteria)


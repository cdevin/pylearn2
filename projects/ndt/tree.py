from pylearn2.utils import serial
from pylearn2.config import yaml_parse
from theano import function
import numpy as np

def load_model(model_path):
    model = serial.load(model_path)

    ds = model.dataset_yaml_src
    ds = yaml_parse.load(ds)

    return model, ds


def branch_funbc(model):
    X = model.get_input_space().make_batch_theano()
    y = model.fprop(X)
    y = y.argmax(axis=1)
    return function([X], y)

def branch_data(brancher, data, batch_size = 100):

    res = []
    for i in xrange(data.shape[0] / batch_size):
        res.append(brancher(ds.X[i * batch_size : (i+1) * batch_size, :]))
    rem = np.mod(data.shape[0], batch_size)
    if rem != 0:
        res.append(brancher(ds.X[data.shape[0] - rem, :]))


    res = np.concatenate(res)
    pos = np.nonzero(res)
    neg = np.where(res == 0)
    return pos, neg



if __name__ == "__main__":

    model_path = 'exp/mnist.pkl'
    model, ds = load_model(model_path)
    brancher = branch_funbc(model)
    pos, neg = branch_data(brancher, ds.X)
    import ipdb
    ipdb.set_trace()


# have custom dataset that accept index

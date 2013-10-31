import numpy as np
from theano import function
import theano.tensor as T
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys

#DATA_PATH = "results/maxout/"
#DATA_PATH = "results/maxout_data/"
DATA_PATH = "results/maxout_class2/"

def get_func(model_path):
    model = serial.load(model_path)
    Xb = model.get_input_space().make_batch_theano()
    Xb.name = 'Xb'

    ymf = model.fprop(Xb)
    ymf.name = 'ymf'
    yl = T.argmax(ymf,axis=1)

    return function([Xb],yl)



def predict(model, data):
    func = get_func(model)
    ds = serial.load(data)
    y_ = func(ds.X)
    return np.argmax(ds.y, axis=1), y_


if __name__ == "__main__":

    #items = [4,5,6,7]
    items = [0,1,2,3]
    labels = []
    pred = []
    for item in items:
        y, y_ = predict("{}mnist{}_best.pkl".format(DATA_PATH, item), "{}test_{}.pkl".format(DATA_PATH, item))
        labels.append(y)
        pred.append(y_)


    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    assert labels.shape[0] == pred.shape[0]
    assert pred.shape[0] == 10000

    print (pred != labels).sum()

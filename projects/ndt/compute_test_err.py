import numpy as np
from theano import function
import theano.tensor as T
from pylearn2.utils import py_integer_types
from pylearn2.utils import serial
from pylearn2.config import yaml_parse
import sys
from sklearn.metrics import confusion_matrix
import pylab as pl

#DATA_PATH = "results/maxout/"
#DATA_PATH = "results/maxout_data/"
DATA_PATH = "results/maxout_class2/"
DATA_PATH = "/RQexec/mirzameh/results/tree/cifar10/maxout0/"

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

def predict_c01b(model, data, batch_size = 500):
    func = get_func(model)
    ds = serial.load(data)
    x = ds.get_topological_view().astype('float32')
    y_ = []
    for i in xrange(x.shape[3] / batch_size):
        y_.append(func(x[:,:,:,i*batch_size : (i+1) * batch_size]))
    rem = np.mod(x.shape[3], batch_size)
    if rem != 0:
        y_.append(func(x[:,:,:,x.shape[3] - rem:]))

    y_ = np.concatenate(y_)
    return np.argmax(ds.y, axis=1), y_




def do_mnist():
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

def do_cifar():
    items = [2,3]
    labels = []
    pred = []
    for item in items:
        y, y_ = predict_c01b("{}{}_best.pkl".format(DATA_PATH, item), "{}test_{}.pkl".format(DATA_PATH, item))
        labels.append(y)
        pred.append(y_)

    cm = confusion_matrix(labels[0], pred[0])
    print cm
    #pl.matshow(cm)
    #pl.title('Confusion matrix')
    #pl.colorbar()
    #pl.ylabel('True label')
    #pl.xlabel('Predicted label')
    #pl.savefig('cm1.png')
    #pl.close()


    cm = confusion_matrix(labels[1], pred[1])
    print cm
    import ipdb
    ipdb.set_trace()
    #pl.matshow(cm)
    #pl.title('Confusion matrix')
    #pl.colorbar()
    #pl.ylabel('True label')
    #pl.xlabel('Predicted label')
    #pl.savefig('cm2.png')
    #pl.close()



    labels = np.concatenate(labels)
    pred = np.concatenate(pred)
    assert labels.shape[0] == pred.shape[0]
    assert pred.shape[0] == 10000

    errors = (pred != labels).sum()
    print errors, labels.shape[0], pred.shape[0]
    print errors / float(labels.shape[0]) *  100
    cm = confusion_matrix(labels, pred)
    print cm

    #pl.matshow(cm)
    #pl.title('Confusion matrix')
    #pl.colorbar()
    #pl.ylabel('True label')
    #pl.xlabel('Predicted label')
    #pl.savefig('cm.png')


if __name__ == "__main__":
    do_cifar()

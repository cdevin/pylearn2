import copy
import numpy as np
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial
import ipdb

DATA_PATH = "results/maxout_class2/"
#DATA_PATH = "results/maxout_data/"

# split datasts in 4 groups such that, each have balanced number of examples per class
def class_balanced(ds, num = 4):

    divs = []
    for digit in range(10):
        index  = np.where(np.argmax(ds.y,axis=1) == digit)[0]
        np.random.shuffle(index)
        data = []
        for item in range(num):
            data.append(index[item::num])
        divs.append(data)


    rval = []
    for item in range(num):
        tmp = []
        for digit in range(10):
            tmp.append(divs[digit][item])

        rval.append(np.concatenate(tmp))

    return rval

def class_balanced2(ds, num = 4):

    rval = []

    index = []
    index.append(np.where(np.argmax(ds.y,axis=1) == 0)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 1)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 2)[0])
    index = np.concatenate(index)
    np.random.shuffle(index)
    rval.append(index)

    index = []
    index.append(np.where(np.argmax(ds.y,axis=1) == 3)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 4)[0])
    index = np.concatenate(index)
    np.random.shuffle(index)
    rval.append(index)

    index = []
    index.append(np.where(np.argmax(ds.y,axis=1) == 5)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 6)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 7)[0])
    index = np.concatenate(index)
    np.random.shuffle(index)
    rval.append(index)

    index = []
    index.append(np.where(np.argmax(ds.y,axis=1) == 8)[0])
    index.append(np.where(np.argmax(ds.y,axis=1) == 9)[0])
    index = np.concatenate(index)
    np.random.shuffle(index)
    rval.append(index)




    return rval


def data_balanced(ds, num=4):

    index = range(ds.y.shape[0])
    np.random.shuffle(index)
    rval = []
    for item in range(num):
        rval.append(index[item::num])

    return rval

def save(ds, divs, name):

    for i, item in enumerate(divs):
        myds = copy.deepcopy(ds)
        myds.X = myds.X[item]
        myds.y = myds.y[item]
        serial.save("{}{}_{}.pkl".format(DATA_PATH, name, i), myds)

# split datasts in 4 groups such that, each have missing some classed


if __name__ == "__main__":
    ds = MNIST('test', one_hot=True)
    divs = class_balanced2(ds)
    save(ds, divs, 'test')

    ds = MNIST('train', one_hot=True, start=0, stop=50000)
    divs = class_balanced2(ds)
    save(ds, divs, 'train')

    ds = MNIST('train', one_hot=True, start=50000, stop=60000)
    divs = class_balanced2(ds)
    save(ds, divs, 'valid')


    #ds = MNIST('test', one_hot=True)
    #divs = data_balanced(ds)
    #save(ds, divs, 'test')

    #ds = MNIST('train', one_hot=True, start=0, stop=50000)
    #divs = data_balanced(ds)
    #save(ds, divs, 'train')

    #ds = MNIST('train', one_hot=True, start=50000, stop=60000)
    #divs = data_balanced(ds)
    #save(ds, divs, 'valid')


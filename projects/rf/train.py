import theano
from theano import tensor
from pylearn2.models.mlp import RectifiedLinear
from ig2 import MLP_IG
from logistic_sgd import load_data
from collections import OrderedDict
import numpy as np

def train():

    n_epochs = 10
    learning_rate = 0.01
    batch_size = 100

    # load data
    dataset = load_data('/data/lisa/exp/mirzamom/DeepLearningTutorials/data/mnist.pkl.gz')
    train_x, train_y = dataset[0]
    n_train_batches = train_x.get_value(borrow=True).shape[0] / batch_size

    # load model
    layer1 = RectifiedLinear(dim = 100, layer_name = 'l1', irange = 0.05)
    layer2 = RectifiedLinear(dim = 100, layer_name = 'l2', irange = 0.05)
    model = MLP_IG(layers = [layer1, layer2], nvis = 784)


    x = tensor.matrix()
    y = tensor.matrix()
    x.tag.test_value = np.random.rand(5, 784).astype('float32')
    y.tag.test_value = np.random.rand(5, 10).astype('float32')
    index = tensor.lscalar()
    grads = model.get_grads((x, y))
    params =  model.get_params()

    updates = OrderedDict()
    for item in params:
        updates[item] = item - learning_rate * grads[item]


    train_model = theano.function(inputs = [index], updates =updates,
            givens = {x: train_x[index * batch_size: (index + 1) * batch_size],
                        y: train_y[index * batch_size: (index + 1) * batch_size]})



    epoch = 0
    while (epoch < n_epochs):
        for minibatch_index in xrnage(n_train_batches):
            train_model(minibatch_index)
        print epoch
        epoch += 1


if __name__ == "__main__":
    train()

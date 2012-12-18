from lcn import lcn
from theano import tensor
import theano
import numpy
import scipy.ndimage
import Image


def apply_lcn(data, topo):

    x = tensor.matrix()
    f = theano.function(inputs=[x],outputs=lcn(x.reshape((1,48,48)),(48,48)))

    orig_shape = data.shape
    data = numpy.concatenate([f(item)for item in data.reshape(topo)])
    return data.reshape(orig_shape)


def reflect(X, Y, topo, seed = 1322):

    orig_shape = X.shape
    X = X.reshape(topo)
    X = numpy.vstack((X, X[:,:,::-1])).reshape(orig_shape[0] *2, orig_shape[1])
    Y = numpy.concatenate((Y, Y))

    return X, Y


def shuffle(X, Y, rng):

    rand_idx = rng.permutation(X.shape[0])
    X = X[rand_idx]
    Y = Y[rand_idx]
    return X, Y

def corner_shuffle(X, Y, topo, rng):


    def corner(img):
        rnd = lambda: rng.rand() * 5.
        new_cord = [rnd(), rnd(),
                    rnd(), 48-rnd(),
                    48-rnd(), 48-rnd(),
                    48-rnd(), rnd()]
        img = Image.fromarray(img)
        img = img.transform((48, 48), Image.QUAD, new_cord, Image.BILINEAR)
        return numpy.asarray(img)

    orig_shape = X.shape
    X = X.reshape(topo)
    new_x = []
    for item in X:
        new_x.append(corner(item))
    X = numpy.vstack((X, numpy.array(new_x))).reshape(orig_shape[0] * 2, orig_shape[1])
    Y = numpy.concatenate((Y, Y))

    return X, Y

def sigmoid_disortion(X, rng):

    const = rng.normal(scale = std, size = X.shape[0])


def rotate(X, topo, rng, angle_limit = 10):

    angles = range(-angle_limit, angle_limit)
    y_list = rng.randint(0, len(angles), X.shape[0])

    orig_shape = X.shape
    X = X.reshape(topo)
    new_x = []
    for item, y in zip(X, y_list):
        new_x.append(scipy.ndimage.interpolation.rotate(item, angles[y], reshape = False).ravel())

    new_x = numpy.vstack(new_x)
    return new_x.reshape(orig_shape), y_list.astype('int32')

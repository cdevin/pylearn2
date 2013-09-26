import numpy
import theano
import theano.tensor as T
from pylearn2.datasets import preprocessing
from pylearn2.linear.conv2d import Conv2D
from pylearn2.space import Conv2DSpace
from pylearn2.utils import sharedX


class LeCunLCN_ICPR(preprocessing.ExamplewisePreprocessor):

    def __init__(self, img_shape, eps=1e-12):
        self.img_shape = img_shape
        self.eps = eps

    def apply(self, dataset, can_fit = True):
        x = dataset.get_topological_view()

        # lcn on y channel of yuv
        x = rgb_yuv(x)
        x[:,:,:,0] = lecun_lcn(x[:,:,:,0], self.img_shape, 7)

        # lcn on each rgb channel
        x = yuv_rgb(x)
        for i in xrange(3):
            x[:,:,:,i] = lecun_lcn(x[:,:,:,i], self.img_shape, 7)

        dataset.set_topological_view(x)

def lecun_lcn(input, img_shape, kernel_shape):

        input = input.reshape(input.shape[0], input.shape[1], input.shape[2], 1)
        X = T.matrix(dtype=input.dtype)
        X = X.reshape((len(input), img_shape[0], img_shape[1], 1))

        filter_shape = (1, 1, kernel_shape, kernel_shape)
        filters = sharedX(gaussian_filter(kernel_shape).reshape(filter_shape))

        input_space = Conv2DSpace(shape = img_shape, num_channels = 1)
        transformer = Conv2D(filters = filters, batch_size = len(input),
                            input_space = input_space,
                            border_mode = 'full')
        convout = transformer.lmul(X)

        # For each pixel, remove mean of 9x9 neighborhood
        mid = int(numpy.floor(kernel_shape/ 2.))
        centered_X = X - convout[:,mid:-mid,mid:-mid,:]

        # Scale down norm of 9x9 patch if norm is bigger than 1
        transformer = Conv2D(filters = filters, batch_size = len(input),
                            input_space = input_space,
                            border_mode = 'full')
        sum_sqr_XX = transformer.lmul(X**2)

        denom = T.sqrt(sum_sqr_XX[:,mid:-mid,mid:-mid,:])
        per_img_mean = denom.mean(axis = [1,2])
        divisor = T.largest(per_img_mean.dimshuffle(0,'x', 'x', 1), denom)

        new_X = centered_X / divisor
        new_X = T.flatten(new_X, outdim=3)

        f = theano.function([X], new_X)
        return f(input)

def rgb_yuv(x):
    r = x[:,:,:,0]
    g = x[:,:,:,1]
    b = x[:,:,:,2]

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g  - 0.10001 * b

    x[:,:,:,0] = y
    x[:,:,:,1] = u
    x[:,:,:,2] = b

    return x

def yuv_rgb(x):
    y = x[:,:,:,0]
    u = x[:,:,:,1]
    v = x[:,:,:,2]

    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u

    x[:,:,:,0] = r
    x[:,:,:,1] = g
    x[:,:,:,2] = b

    return x

def gaussian_filter(kernel_shape):
    x = numpy.zeros((kernel_shape, kernel_shape), dtype='float32')

    def gauss(x, y, sigma=2.0):
        Z = 2 * numpy.pi * sigma**2
        return  1./Z * numpy.exp(-(x**2 + y**2) / (2. * sigma**2))

    mid = numpy.floor(kernel_shape/ 2.)
    for i in xrange(0,kernel_shape):
        for j in xrange(0,kernel_shape):
            x[i,j] = gauss(i-mid, j-mid)

    return x / numpy.sum(x)




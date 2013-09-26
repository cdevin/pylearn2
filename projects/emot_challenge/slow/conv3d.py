"""
The functionality in this module is very similar to that in
pylearn2.linear.conv2d. The difference is that this module is
based on Alex Krizhevsky's cuda-convnet convolution, while
pylearn2.linear.conv2d is based on theano's 2D convolution.
This module therefore uses the axis format ('c', 0, 1, 'b')
as its native format, while the other uses ('b', 'c', 0, 1).
This module also requires the use of GPU, while the other
supports CPU.
"""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "Ian Goodfellow"
__email__ = "goodfeli@iro"

import functools
import numpy as np
import warnings

from theano.sandbox import cuda
import theano.tensor as T

if cuda.cuda_enabled:
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
    from theano.sandbox.cuda import gpu_from_host
    from theano.sandbox.cuda import host_from_gpu

from pylearn2.utils import sharedX
from pylearn2.linear.conv2d import default_rng
from pylearn2.linear.conv2d import default_sparse_rng
from pylearn2.linear.linear_transform import LinearTransform
from pylearn2.sandbox.cuda_convnet import check_cuda
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.filter_acts import ImageActs
from pylearn2.packaged_dependencies.theanoconv3d2d.conv3d2d import conv3d
from noisy_encoder.scripts.emot_challenge.space import Conv3DSpace
#from

class Conv3D(LinearTransform):
    """
    A pylearn2 linear operator based on 3D convolution,
    implemented using Alex Krizhevsky's cuda-convnet library.

    """

    def __init__(self,
            filters,
            input_axes = ('b', 't', 'c', 0, 1),
            batch_size=None,
            output_axes = ('b', 't', 'c', 0, 1),
            kernel_stride = (1, 1, 1)):

        if len(kernel_stride) != 3:
            raise ValueError("kernel_stride must have length 3")

        if batch_size != None:
            raise NotImplementedError()

        self.input_axes = input_axes
        self.output_axes = output_axes

        self._filters = filters
        self._kernel_stride = kernel_stride

    @functools.wraps(LinearTransform.get_params)
    def get_params(self):
        return [self._filters]

    @functools.wraps(LinearTransform.get_weights_topo)
    def get_weights_topo(self, borrow=False):
        inp, rows, cols, outp = range(4)
        raw = self._filters.get_value(borrow=borrow)
        return np.transpose(raw, (outp, rows, cols, inp))

    def lmul(self, x):
        """
        dot(x, A)
        aka, do convolution with input image x

        """

        check_cuda(str(type(self)) + ".lmul")

        cpu = 'Cuda' not in str(type(x))

        if cpu:
            x = gpu_from_host(x)

        # x must be formatted as channel, topo dim 0, topo dim 1, batch_index
        # for use with FilterActs
        assert x.ndim == 5
        x_axes = self.input_axes
        assert len(x_axes) == 5

        op_axes = ('b', 't', 'c', 0, 1)

        if tuple(x_axes) != op_axes:
            x = x.dimshuffle(*[x_axes.index(axis) for axis in op_axes])

        rval = conv3d(x, self._filters,
                        signals_shape = x.shape,
                        filters_shape = self._filters.shape,
                        subsample = self._kernel_stride,
                        border_mode = 'valid')


        # Format the output based on the output space
        rval_axes = self.output_axes
        assert len(rval_axes) == 5

        if tuple(rval_axes) != op_axes:
            rval = rval.dimshuffle(*[op_axes.index(axis) for axis in rval_axes])

        if cpu:
            rval = host_from_gpu(rval)

        return rval

    def set_batch_size(self, batch_size):
        pass

def make_random_conv3D(irange, input_axes, output_axes, filter_shape,  kernel_stride = (1,1,1), rng = None):

    if rng is None:
        rng = default_rng()

    W = sharedX(rng.uniform(-irange,irange,(filter_shape)))

    return Conv3D(filters = W,
        input_axes = input_axes,
        output_axes = output_axes,
        kernel_stride = kernel_stride)

def setup_detector_layer(layer, input_space, rng, irange):
    """
    Takes steps to set up an object for use as being some kind of convolutional layer.
    This function sets up only the detector layer.

    Parameters
    ----------
    layer: Any python object that allows the modifications described below and has
        the following attributes:
            pad: int describing amount of zero padding to add
            kernel_shape: 2-element tuple or list describing spatial shape of kernel
            fix_kernel_shape: bool, if true, will shrink the kernel shape to make it
                feasible, as needed (useful for hyperparameter searchers)
            detector_channels: The number of channels in the detector layer
            init_bias: A numeric constant added to a tensor of zeros to initialize the
                    bias
            tied_b: If true, biases are shared across all spatial locations

    input_space: A Conv2DSpace to be used as input to the layer

    rng: a numpy RandomState or equivalent

    irange: float. kernel elements are initialized randomly from U(-irange, irange)

    Does the following:
        raises a RuntimeError if cuda is not available
        sets layer.input_space to input_space
        sets up addition of dummy channels for compatibility with cuda-convnet:
            layer.dummy_channels: # of dummy channels that need to be added
                (You might want to check this and raise an Exception if it's not 0)
            layer.dummy_space: The Conv2DSpace representing the input with dummy channels
                added
        sets layer.detector_space to the space for the detector layer
        sets layer.transformer to be a Conv2D instance
        sets layer.b to the right value


        TODO: Move me
        Input sahape : batch size, times shape, channels, rows, columns
        Filter shape: Output channels, filter time, input channels, filter row, filter col


       On thenoConv3d2d:
       Ns : batch size
       Hs: 0
       Ws: 1
       Nf: output channels
       Tf: filter times
       Ts: input time
       C: input channels
    """

    # Use "self" to refer to layer from now on, so we can pretend we're just running
    # in the set_input_space method of the layer
    self = layer

    # Validate input
    if not isinstance(input_space, Conv3DSpace):
        raise TypeError("The input to a convolutional layer should be a Conv3DSpace, "
                " but layer " + self.layer_name + " got "+str(type(self.input_space)))

    if not hasattr(self, 'detector_channels'):
        raise ValueError('layer argument must have a "detector_channels" attribute specifying how many channels to put in the convolution kernel stack.')

    # Store the input space
    self.input_space = input_space

    if hasattr(self, 'kernel_stride'):
        kernel_stride = self.kernel_stride
    else:
        kernel_stride = [1, 1, 1]


    output_shape = [int(np.ceil((i_sh + 2. * self.pad - k_sh) / float(k_st))) + 1
                        for i_sh, k_sh, k_st in zip(self.input_space.shape,
                                                    self.kernel_shape,
                                                    kernel_stride)]

    output_sequence_length = self.input_space.sequence_length - self.kernel_sequence_length + 1

    def handle_kernel_shape(idx):
        if self.kernel_shape[idx] < 1:
            raise ValueError("kernel must have strictly positive size on all axes but has shape: "+str(self.kernel_shape))
        if output_shape[idx] <= 0:
            if self.fix_kernel_shape:
                self.kernel_shape[idx] = self.input_space.shape[idx] + 2 * self.pad
                assert self.kernel_shape[idx] != 0
                output_shape[idx] = 1
                warnings.warn("Had to change the kernel shape to make network feasible")
            else:
                raise ValueError("kernel too big for input (even with zero padding)")

    map(handle_kernel_shape, [0, 1])

    # batch, 0, 1, t, c

    self.detector_space = Conv3DSpace(shape=output_shape,
                                      sequence_length = output_sequence_length,
                                      num_channels = self.detector_channels,
                                      axes = ('b', 't', 'c', 0, 1))

    filter_shape = (self.detector_channels,
                        self.kernel_sequence_length,
                        self.input_space.num_channels,
                        self.kernel_shape[0],
                        self.kernel_shape[1])

    import ipdb
    ipdb.set_trace()
    self.transformer = make_random_conv3D(
          irange = self.irange,
          input_axes = self.input_space.axes,
          output_axes = self.detector_space.axes,
          filter_shape = filter_shape,
          kernel_stride = kernel_stride,
          rng = rng)

    W, = self.transformer.get_params()
    W.name = 'W'

    if self.tied_b:
        self.b = sharedX(np.zeros((self.detector_space.num_channels)) + self.init_bias)
    else:
        self.b = sharedX(self.detector_space.get_origin() + self.init_bias)
    self.b.name = 'b'

    print 'Input shape: ', self.input_space.shape
    print 'Detector space: ', self.detector_space.shape


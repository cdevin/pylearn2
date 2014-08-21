"""
Layers that take Sequences as inputs but are not recurrent.
"""

from functools import wraps
import numpy as np
from theano import tensor
from pylearn2.models.mlp import Layer
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.sandbox.rnn.space import SequenceSpace
from pylearn2.utils import sharedX
from theano import config, scan
from theano.compat.python2x import OrderedDict
#from pylearn2.sandbox.rnn.models.rnn import Recurrent

class BagOfParts(Layer):
    """ A layer that takes a sequence of vectors, sums them , and passes the 
        result though a linear layer and a nonlinearity.

    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    irange : float
        The input-to-hidden weight matrix is initialized with weights in
        the uniform interval (-irange, irange). The hidden-to-hidden
        matrix weights are sampled in the same manner, unless the argument
        svd is set to True (see below).
    init_bias : float, optional
        Set an initial bias to be added at each time step. Defaults to 0.
    nonlinearity : theano function, optional
        Defaults to tensor.tanh, the non-linearity to be applied to the
        hidden state after each update
    """
    
    def __init__(self, dim, layer_name, irange, init_bias,
                 nonlinearity=tensor.tanh):
                 
        self.rnn_friendly = True
        self.__dict__.update(locals())
        del self.self

        super(BagOfParts, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space
        self.output_space = VectorSpace(dim=self.dim)
        
        # Initialize the parameters
        rng = self.mlp.rng
        if self.irange is None:
            raise ValueError("Recurrent layer requires an irange value in "
                             "order to initialize its weight matrices")

        # W is the input-to-hidden matrix
        W = rng.uniform(-self.irange, self.irange,
                        (self.input_space.dim, self.dim))

        self._params = [sharedX(W, name=(self.layer_name + '_W')),
                        sharedX(np.zeros(self.dim) + self.init_bias,
                                name=self.layer_name + '_b')]

    def fprop(self, state_below, mask=None):
        W, b = self._params
        
        sum_ = TT.sum(state_below, axis=1)
        rval = self.nonlinearity(TT.dot(sum_, W) + b)
        return rval
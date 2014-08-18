"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import theano.tensor as T
from theano import config

from pylearn2.models import mlp
from pylearn2.models.mlp import Layer
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul

from theano.compat.python2x import OrderedDict
from pylearn2.sandbox.rnn.space import SequenceSpace
from theano import scan
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams


class Softmax(mlp.Softmax):
    """
    An extension of the MLP's softmax layer which monitors
    the perplexity

    Parameters
    ----------
    n_classes : WRITEME
    layer_name : WRITEME
    irange : WRITEME
    istdev : WRITEME
    sparse_init : WRITEME
    W_lr_scale : WRITEME
    b_lr_scale : WRITEME
    max_row_norm : WRITEME
    no_affine : WRITEME
    max_col_norm : WRITEME
    init_bias_target_marginals : WRITEME
    """
    @wraps(Layer.get_monitoring_channels_from_state)
    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval = OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())])

        if target is not None:
            y_hat = T.argmax(state, axis=1)
            y = T.argmax(target, axis=1)
            misclass = T.neq(y, y_hat).mean()
            misclass = T.cast(misclass, config.floatX)
            rval['misclass'] = misclass
            rval['nll'] = self.cost(Y_hat=state, Y=target)
            rval['ppl'] = 2 ** (rval['nll'] / T.log(2))

        return rval


class ProjectionLayer(Layer):
    """
    This layer can be used to project discrete labels into a continous space
    as done in e.g. language models. It takes labels as an input (IndexSpace)
    and maps them to their continous embeddings and concatenates them.

    Parameters
        ----------
    dim : int
        The dimension of the embeddings. Note that this means that the
        output dimension is (dim * number of input labels)
    layer_name : string
        Layer name
    irange : numeric
       The range of the uniform distribution used to initialize the
       embeddings. Can't be used with istdev.
    istdev : numeric
        The standard deviation of the normal distribution used to
        initialize the embeddings. Can't be used with irange.
    """
    def __init__(self, dim, layer_name, irange=None, istdev=None):
        """
        Initializes a projection layer.
        """
        super(ProjectionLayer, self).__init__()
        self.dim = dim
        self.layer_name = layer_name
        if irange is None and istdev is None:
            raise ValueError("ProjectionLayer needs either irange or"
                             "istdev in order to intitalize the projections.")
        elif irange is not None and istdev is not None:
            raise ValueError("ProjectionLayer was passed both irange and "
                             "istdev but needs only one")
        else:
            self._irange = irange
            self._istdev = istdev

    @wraps(Layer.get_monitoring_channels)
    def get_monitoring_channels(self):

        W, = self.transformer.get_params()

        assert W.ndim == 2

        sq_W = T.sqr(W)

        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))

        return OrderedDict([('row_norms_min',  row_norms.min()),
                            ('row_norms_mean', row_norms.mean()),
                            ('row_norms_max',  row_norms.max()),
                            ('col_norms_min',  col_norms.min()),
                            ('col_norms_mean', col_norms.mean()),
                            ('col_norms_max',  col_norms.max()), ])

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if isinstance(space, IndexSpace):
            self.input_dim = space.dim
            self.input_space = space
        else:
            raise ValueError("ProjectionLayer needs an IndexSpace as input")
        self.output_space = VectorSpace(self.dim * self.input_dim)
        rng = self.mlp.rng
        if self._irange is not None:
            W = rng.uniform(-self._irange,
                            self._irange,
                            (space.max_labels, self.dim))
        else:
            W = rng.randn(space.max_labels, self.dim) * self._istdev

        W = sharedX(W)
        W.name = self.layer_name + '_W'

        self.transformer = MatrixMul(W)

        W, = self.transformer.get_params()
        assert W.name is not None

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        z = self.transformer.project(state_below)
        return z

    @wraps(Layer.get_params)
    def get_params(self):
        W, = self.transformer.get_params()
        assert W.name is not None
        params = [W]
        return params

class PartialBag(Layer):
    """
    A recurrent neural network layer using the hyperbolic tangent
    activation function, passing on all hidden states or a selection
    of them to the next layer.

    The hidden state is initialized to zeros.

    Parameters
    ----------
    dim : int
        The number of elements in the hidden layer
    layer_name : str
        The name of the layer. All layers in an MLP must have a unique name.
    """
    def __init__(self, dim, layer_name):
        self.dim = 3*dim
        self.__dict__.update(locals())
        self.rnn_friendly = True
        del self.self
        super(PartialBag, self).__init__()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        if (not isinstance(space, SequenceSpace) or
                not isinstance(space.space, VectorSpace)):
            raise ValueError("Recurrent layer needs a SequenceSpace("
                             "VectorSpace) as input but received  %s instead"
                             % (space))
        self.input_space = space
        
        self.output_space = VectorSpace(dim=self.dim*3)

    @wraps(Layer.get_params)
    def get_params(self):
        return []

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # TODO: if word is 1 char long, don't double count it.
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This code is incorrect"
        print "It double counts the character of a single character word"
        state_below, mask = state_below
        shape = state_below.shape
        range_ = T.arange(shape[1])
        first_chars = state_below[0, range_]
        last_indices = T.sum(mask, axis=1, dtype='int64') -1
        last_chars = state_below[last_indices, range_]
        #last_chars = state_below[range_, last_indices]
        middle_chars = T.sum(state_below, axis=1) - first_chars - last_chars
        # out0 = T.alloc(0., (shape[0], shape[2]))
        # def fprop_step(state_below, last_index, out):
        #     middle_chars = T.sum(state_below[1:last_index], axis=1)
        #     print "state_below", state_below.ndim
        #     print "middle", middle_chars.ndim
        #     #last_chars = state_below[last_index]
        #     return middle_chars#, last_chars

        # middle_chars, updates = scan(fn=fprop_step, sequences=[state_below, last_indices],
        #                              outputs_info=[out0]
        #                          )

        rval = middle_chars # T.concatenate((first_chars.T, middle_chars.T, last_chars.T), axis=0)
        return rval

class NoisySoftmax(Softmax):
    """
    An extension of the Softmax layer which computes noisy contrastive cost
    """
    def _cost(self, Y, Y_hat):

        assert hasattr(Y_hat, 'owner')
        owner = Y_hat.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
           assert len(owner.inputs) == 1
           Y_hat, = owner.inputs
           owner = Y_hat.owner
           op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2

        assert self._has_binary_target
        assert not self.no_affine

        theano_rng = MRG_RandomStreams(max(self.mlp.rng.randint(2 ** 15), 1))

        # state_below
        state_below = z.owner.inputs[0].owner.inputs[0]

        def _grab_probs(W, b, state_below, S):
            S = S.flatten()
            return T.nnet.sigmoid((W[:,S].T * state_below).sum(1) + b[S])

        cost_pos = T.log(_grab_probs(self.W, self.b, state_below, Y))
        # a single negative sample for each training sample
        Y_neg = T.cast(theano_rng.uniform(state_below.shape[0].reshape([1])) * self.W.shape[1], 'int64')
        cost_neg = T.cast(state_below.shape[0], 'float32') * \
                   T.log(1. - _grab_probs(self.W, self.b, state_below, Y_neg))

        log_prob_of = cost_pos + cost_neg

        return log_prob_of
        

    @wraps(Layer.cost)
    def cost(self, Y, Y_hat):

        log_prob_of = self._cost(Y, Y_hat)
        rval = log_prob_of.mean()
        return - rval

    @wraps(Layer.cost_matrix)
    def cost_matrix(self, Y, Y_hat):
        log_prob_of = self._cost(Y, Y_hat)
        return -log_prob_of






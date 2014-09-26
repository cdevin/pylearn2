"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import numpy as np
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

from theano.compat.python2x import OrderedDict
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
    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None, state=None,
                                      target=None):

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

class NoisySoftmax(Softmax):
    """
    An extension of the Softmax layer which computes noisy contrastive cost
    """
    def __init__(self, exact_q=False, **kwargs):
        super(NoisySoftmax, self).__init__(**kwargs)

        # use the exact distribution to sample from
        self.exact_q = exact_q


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
        if self.exact_q:
            Y_neg = theano_rng.multinomial(pvals=Y_hat).argmax(-1)
        else:
            Y_neg = T.cast(theano_rng.uniform(state_below.shape[0].reshape([1])) * self.W.shape[1], 'int64')
        cost_neg = T.cast(state_below.shape[0], 'float32') * T.log(1. - _grab_probs(self.W, self.b, state_below, Y_neg))

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

class HierarchicalSoftmax(Layer):
    """
    A GPU implementation of the hierarchichal softmax. Uses a two level tree.

    The term 'word' refers to the leaves of the tree and 'group'
    refers to the non-leaf nodes of the tree. The term 'class' will not
    be used because in the plain Softmax layer, class refers to the number
    of outputs, but in many hierarchical softmax implementations, class
    refers to the number of non-leaf nodes.


    Parameters
    ----------
    n_words : int
        Number of possible outputs for softmax targets.
    layer_name : string
        Name of Softmax layers.
    irange : float
        If specified, initialized each weight randomly in 
        U(-irange, irange).
    istdev : float
        If specified, initialize each weight randomly from
        N(0,istdev).
    no_affine : boolean
        If True, softmax nonlinearity is applied directly to
        inputs.
    binary_target_dim : int, optional
        If your targets are class labels (i.e. a binary vector) then set the
        number of targets here so that an IndexSpace of the proper dimension
        can be used as the target space. This allows the softmax to compute
        the cost much more quickly than if it needs to convert the targets
        into a VectorSpace.
    full_softmax: bool,  defaults to false
        Set to true if you want to calculate the probablility of every word.
    """

    def __init__(self, n_words, layer_name, irange=None,
                 istdev=None, no_affine=False,
                 binary_target_dim=None, full_softmax=False):

        super(Softmax, self).__init__()

        self.__dict__.update(locals())
        del self.self
        del self.init_bias_target_marginals

        assert isinstance(n_classes, py_integer_types)

        if binary_target_dim is not None:
            assert isinstance(binary_target_dim, py_integer_types)
            self._has_binary_target = True
            self._target_space = IndexSpace(dim=binary_target_dim, 
                                            max_labels=n_classes)
        else:
            self._has_binary_target = False
    
        self.output_space = VectorSpace(n_classes)

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_words
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space

        rng = self.mlp.rng
        self.group_size = np.ceil(np.sqrt(self.n_words)).astype('int64')
        self.n_groups = np.ceil(self.n_words/float(self.group_size)).astype('int64')
        self.iBlocks = 1  # number of blocks in the input (from lower layer)

        if self.no_affine:
            self._params = []
        else:
            if self.irange is not None:
                assert self.istdev is None
                W1 = rng.uniform(-self.irange,
                                 self.irange,
                                 (self.input_dim, self.n_groups))

                # Has an input-> word matrix for each group
                W2 = rng.uniform(-self.irange,
                                 self.irange, 
                                 (self.iBlocks, self.n_groups, 
                                  self.input_dim, self.group_size))
            elif self.istdev is not None:
                W1 = rng.randn(self.input_dim, self.n_groups) * self.istdev
                W2 = rng.randn(self.iBlocks, self.n_groups, self.input_dim, 
                               self.group_size) * self.istdev

            self.W1 = sharedX(W1,  '%s_W1'%self.layer_name)
            self.b1 = sharedX(np.zeros((self.n_groups,)), name='%s_b1'%self.layer_name)
            self.W2 = sharedX(W2,  '%s_W2'%self.layer_name)
            self.b1 = sharedX(np.zeros((self.n_groups, self.group_size)), 
                              name='%s_b1'%self.layer_name)

            self._params = [self.b1, self.W1, self.b2, self.W2]
            

    @wraps(Layer.fprop)
    def fprop(self, state_below):
        self.input_space.validate(state_below)
        if self.needs_reformat:
            state_below = self.input_space.format_as(state_below,
                                                     self.desired_space)

        self.desired_space.validate(state_below)
        assert state_below.ndim == 2

        if not hasattr(self, 'no_affine'):
            self.no_affine = False

        # if self.no_affine:
        #     group_vals = state_below;
        # else:
        #     assert self.W1.ndim == 2
        #     group_vals = T.dot(state_below, self.W1) + b1


        if self.full_softmax:
            group_vals = T.dot(state_below, self.W1) + b1
            groups_vec = T.arange(self.n_groups)
            
            
            def _compute_ingroup(group_index):
                word_vals = T.nnet.softmax(
                    TT.dot(state_below, self.W2[0, group_index, :, :])
                    +self.b2[group_index,:])
                weighted_word_val = word_val * group_vals[:, group_index][:,None]
                return weighted_word_val.T

            val = theano.scan(_compute_ingroup, 
                              groups_vec, 
                              None, 
                              name='compute_ingroup')
            print val.ndim
            all_word_val = val[0].reshape([val[0].shape[0]*val[0].shape[1], val[0].shape[2]]).T
            all_word_val = all_word_val[:,:self.n_out]
            emb_val = all_word_val

        for value in get_debug_values(rval):
            if self.mlp.batch_size is not None:
                assert value.shape[0] == self.mlp.batch_size

        return rval


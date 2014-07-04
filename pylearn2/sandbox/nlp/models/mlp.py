"""
Sandbox multilayer perceptron layers for natural language processing (NLP)
"""
import theano.tensor as T
from theano import config

from pylearn2.models import mlp
from pylearn2.models.mlp import Layer, Linear
from pylearn2.space import IndexSpace
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX
from pylearn2.utils import wraps
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from theano.compat.python2x import OrderedDict


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

class NCE(Linear):

    def __init__(self, num_samples=5, noise_p = None, max_labels=1, **kwargs):

        super(NCE, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

        self._max_labels = max_labels
        self.num_samples = num_samples
        #self.noise = sharedX(np.zeros(batch_size * self.k), dtype = 'int64')

        if noise_p is None:
            self.uniform = True
            self.noise_p = sharedX(1. / max_labels)
        else:
            self.uniform = False
            self.noise_p = sharedX(noise_p)
            self.theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        self._target_space = IndexSpace(dim=1, max_labels=max_labels)
        self._noise_space = IndexSpace(dim=num_samples, max_labels=max_labels)

    def delta(self, Y, Y_hat, k = 1):

        if Y.ndim != 1:
            Y = Y.flatten().dimshuffle(0)

        if self.uniform is True:
            rval = Y_hat - T.log(self.k * self.noise_p)

        # Not sure what score should be in this case 
        # else:
        #     rval = self.score(X, Y, k = k)
        #     rval = rval - T.log(self.k * self.noise_p[Y]).reshape(rval.shape)
        return T.cast(rval, config.floatX)

    def get_noise(self):

        if self.uniform:
            if self.batch_size is None:
                raise NameError("Since numpy random is faster, batch_size is required")
            #noise = self.rng.randint(0, self.dict_size - 1, self.batch_size * self.k)
            #self.noise.set_value(noise)
            rval = self.theano_rng.random_integers(
                low = 0, 
                high = self.dict_size -1,
                size = (self.batch_size * self.num_samples)
            )
        else:
            #rval = self.rng.multinomial(n = 1, pvals = self.noise_p.get_value(), size = self.batch_size * self.k)
            rval = self.theano_rng.multinomial(pvals = T.repeat(
                self.noise_p.reshape((1, self.dict_size)),
                repeats= self.k * self.batch_size, axis=0)
            )
            # !!! Why argmax here?
            rval = T.argmax(rval, axis=1)
        return rval

    def cost(self, Y, Y_hat):
        noise = self.get_noise()
        pos = T.nnet.sigmoid(self.delta(Y_hat, Y))
        neg = 1. - T.nnet.sigmoid((self.delta(Y_hat, noise, k = self.k)))
        neg = neg.sum(axis=0)

        rval = T.log(pos) + self.k * T.log(neg)
        #rval = T.log(pos) + T.log(neg)
        return -rval.mean()

        
        
        
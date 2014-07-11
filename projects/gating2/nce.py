import numpy as np
from theano import tensor as T
from theano import config
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import Layer,Linear,MLP,Softmax
from pylearn2.monitor import get_monitor_doc
from pylearn2.space import Space,VectorSpace, IndexSpace, CompositeSpace
from pylearn2.format.target_format import OneHotFormatter
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.utils import as_floatX
from pylearn2.costs.mlp import Default

class NCE(Layer):
    def __init__(self, n_classes, layer_name, num_noise_samples = 5,
            noise_prob = None,
            disable_ppl_monitor = False,
            irange=None,
             istdev=None,
             sparse_init=None, W_lr_scale=None,
             b_lr_scale=None, max_row_norm=None,
             no_affine=False,
             max_col_norm=None, init_bias_target_marginals=None):

        super(NCE, self).__init__()

        self.__dict__.update(locals())
        del self.self
        
        self.output_space = IndexSpace(dim=1, max_labels=self.n_classes)
        self.n_classes = n_classes
        self.k = num_noise_samples
        rng = np.random.RandomState(44)
        self.rng = rng

        if noise_prob is None:
            self.uniform = True
            self.noise_p = sharedX(1. / self.dict_size)
        else:
            self.uniform = False
            self.noise_p = sharedX(noise_prob)
            self.theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))
        #self.set_spaces()

    def fprop(self,state_below):
        return state_below

    def get_params(self):
        # rval = self.projector.get_params()
        # rval.extend(self.y_projector.get_params())
        # rval.extend([self.C, self.b])
        return []

    def nll(self, Y_hat, Y):
        z = Y_hat
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        Y = OneHotFormatter(self.n_classes).theano_expr(Y)
        Y = Y.reshape((Y.shape[0], Y.shape[2]))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1
        rval = as_floatX(log_prob_of.mean())
        return - rval
    
    def scoreold(self, Y_hat, Y=None, k = 1):
        q_h = Y_hat
        q_h = self.fprop(X)
        # this is used during training
        if Y is not None:
            if Y.ndim != 1:
                Y = Y.flatten().dimshuffle(0)
            q_w = self.projector.project(Y)
            rval = (q_w.reshape((k, X.shape[0], q_h.shape[1])) * q_h).sum(axis=2)
            rval = rval + self.b[Y].reshape((k, X.shape[0]))
        # during nll
        else:
            q_w = self.y_projector._W
            rval = T.dot(q_h, q_w.T) + self.b.dimshuffle('x', 0)
        return rval

    def score(self, Y_hat, Y, k = 1):        
        ind = T.arange(Y_hat.shape[0])
        tind = T.tile(ind,(k,))
        if Y.ndim != 1:
            Y = Y.flatten().dimshuffle(0)
        rval = Y_hat[tind,Y]
        
        rval = rval.reshape((k,Y_hat.shape[0]))
        return rval
        # yhat 100x10000
        # y 100
        # noisey 100x10

    def delta(self, Y_hat, Y, k = 1):
        if Y.ndim != 1:
            Y = Y.flatten().dimshuffle(0)

        if self.uniform is True:
            rval = self.score(Y_hat, Y,k=k) - T.log(self.k * self.noise_p)
        else:
            rval = self.score(Y_hat, Y, k = k)
            rval = rval - T.log(self.k * self.noise_p[Y]).reshape(rval.shape)
        return T.cast(rval, config.floatX)

    def cost(self, Y, Y_hat, noise):
    	pos = T.nnet.sigmoid(self.delta(Y_hat, Y))
        neg = 1. - T.nnet.sigmoid((self.delta(Y_hat, noise, k = self.k)))
        neg = neg.sum(axis=0)
        rval = T.log(pos) + self.k * T.log(neg)
        return -rval.mean()

    def get_layer_monitoring_channels(self, state_below=None,state=None, targets=None):
    	rval = OrderedDict()
        # channels that does not require state information
        # if self.no_affine:
        #     rval = OrderedDict()

        # W = self.W

        # assert W.ndim == 2

        # sq_W = T.sqr(W)

        # row_norms = T.sqrt(sq_W.sum(axis=1))
        # col_norms = T.sqrt(sq_W.sum(axis=0))

        # rval = OrderedDict([('row_norms_min',  row_norms.min()),
        #                     ('row_norms_mean', row_norms.mean()),
        #                     ('row_norms_max',  row_norms.max()),
        #                     ('col_norms_min',  col_norms.min()),
        #                     ('col_norms_mean', col_norms.mean()),
        #                     ('col_norms_max',  col_norms.max()), ])
	 
        if (state_below is not None) or (state is not None):
            if state is None:
                state = self.fprop(state_below)

            if targets is not None:
                y_hat = T.argmax(state, axis=1)
                y = T.argmax(targets, axis=1)
                misclass = T.neq(y, y_hat).mean()
                misclass = T.cast(misclass, config.floatX)
                rval['misclass'] = misclass
                rval['nll'] = self.nll(Y_hat=state, Y=targets)
                if not self.disable_ppl_monitor:
	                rval['nll'] = self.nll(Y_hat=state, Y=targets)
	                rval['perplexity'] = 10 ** (rval['nll'] / np.log(10)).astype(config.floatX)
	                rval['entropy'] = rval['nll'] / np.log(2).astype(config.floatX)
        return rval

    def set_input_space(self, space):

        self.input_space = space

        if not isinstance(space, Space):
            raise TypeError("Expected Space, got " +
                            str(space)+" of type "+str(type(space)))

        self.input_dim = space.get_total_dimension()
        self.needs_reformat = not isinstance(space, VectorSpace)

        if self.no_affine:
            desired_dim = self.n_classes
            assert self.input_dim == desired_dim
        else:
            desired_dim = self.input_dim
        self.desired_space = VectorSpace(desired_dim)

        if not self.needs_reformat:
            assert self.desired_space == self.input_space
        rng = self.mlp.rng

        # if self.no_affine:
        #     self._params = []
        # else:
        #     if self.irange is not None:
        #         assert self.istdev is None
        #         assert self.sparse_init is None
        #         W = rng.uniform(-self.irange,
        #                         self.irange,
        #                         (self.input_dim, self.n_classes))
        #     elif self.istdev is not None:
        #         assert self.sparse_init is None
        #         W = rng.randn(self.input_dim, self.n_classes) * self.istdev
        #     else:
        #         assert self.sparse_init is not None
        #         W = np.zeros((self.input_dim, self.n_classes))
        #         for i in xrange(self.n_classes):
        #             for j in xrange(self.sparse_init):
        #                 idx = rng.randint(0, self.input_dim)
        #                 while W[idx, i] != 0.:
        #                     idx = rng.randint(0, self.input_dim)
        #                 W[idx, i] = rng.randn()

        #     self.W = sharedX(W,  'softmax_W')

        #     self._params = [self.b, self.W]
    
class MLP_NCE(MLP):
    def __init__(self, k, **kwargs):
        self.k = k

        super(MLP_NCE,self).__init__(**kwargs)
        self.noise_space = IndexSpace(dim=self.k, max_labels=self.layers[-1].n_classes)
        

    def get_monitoring_channels(self, data):
        # if the MLP is the outer MLP \
        # (ie MLP is not contained in another structure)
        X, Y, noise = data
        state = X
        rval = self.get_layer_monitoring_channels(state_below=X,
                                                    targets=Y)
        return rval	
    
    def cost(self, Y, Y_hat, noise):
        return self.layers[-1].cost(Y, Y_hat,noise)


    def cost_from_X(self, data):
        """

        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an
        argument.

        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)

        Parameters
        ----------
        data : WRITEME
        """
        X, Y, noise = data
        assert 'int' in noise.dtype
        self.cost_from_X_data_specs()[0].validate(data)
        
        Y_hat = self.fprop(X)
        return self.cost(Y, Y_hat, noise)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space(),self.get_noise_space()))
        source = (self.get_input_source(), self.get_target_source(),self.get_noise_source())
        return (space, source)

    def get_noise_space(self):
        return self.noise_space

    def get_noise_source(self):
        return 'noises'
        
    def get_monitoring_data_specs(self):
        space = CompositeSpace((self.get_input_space(),
                                 self.get_output_space(),self.get_noise_space()))
        source = (self.get_input_source(), self.get_target_source(),self.get_noise_source())
        return (space, source)

class Cost_noise(Default):
    def get_data_specs(self, model):
        space = CompositeSpace((model.get_input_space(),
                                model.get_output_space(),model.get_noise_space()))
        source = (model.get_input_source(), model.get_target_source(),model.get_noise_source())
        return (space, source)

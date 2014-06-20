import numpy as np
from theano import tensor as T
from theano import config
from theano.printing import Print
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.python2x import OrderedDict
from pylearn2.models.mlp import MLP
from pylearn2.models.mlp import Softmax
from pylearn2.monitor import get_monitor_doc
from pylearn2.space import VectorSpace, IndexSpace, CompositeSpace
from pylearn2.format.target_format import OneHotFormatter
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
from pylearn2.sandbox.nlp.linear.matrixmul import MatrixMul
from pylearn2.models.model import Model
from pylearn2.utils import as_floatX
#import ipdb


class NCE(Softmax):

    def __init__(self,
                num_noise_samples = 2,
                noise_prob = None,
                disable_ppl_monitor = True,
                **kwargs):

        super(NCE, self).__init__(**kwargs)
        self.num_noise_samples = num_noise_samples
        if noise_prob is not None:
            noise_prob = sharedX(noise_prob)
        self.noise_prob = noise_prob
        self.disable_ppl_monitor = disable_ppl_monitor
        #self.output_space = VectorSpace(1)


    def cost(self, Y, Y_hat):
        # TODO fix me later when using IndexSpace

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
        state_below, = owner.inputs
        assert state_below.ndim == 2

        # TODO make this more generic like above
        state_below = state_below.owner.inputs[0].owner.inputs[0]

        Y = T.argmax(Y, axis = 1)
        k = self.num_noise_samples

        if self.noise_prob is None:
            theano_rng = RandomStreams(seed = self.mlp.rng.randint(2 ** 15))
            noise = theano_rng.random_integers(size = (state_below.shape[0], self.num_noise_samples,), low=0, high = self.n_classes - 1)
            p_n = 1. / self.n_classes
            p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])
            p_x = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()])
            # TODO is this reshape necessary?
            p_x = p_x.reshape((state_below.shape[0], k))

            pos = k * p_n / (p_w + k * p_n) * T.log(p_w)
            neg = (p_x / (p_x + k * p_n) * T.log(p_x)).sum(axis=1)
        else:
            #import ipdb
            #ipdb.set_trace()
            theano_rng = MRG_RandomStreams(max(self.mlp.rng.randint(2 ** 15), 1))
            assert self.mlp.batch_size is not None
            noise = theano_rng.multinomial(pvals = np.tile(self.noise_prob.get_value(), (k * self.mlp.batch_size, 1)))
            noise = T.argmax(noise, axis = 1)
            p_n = self.noise_prob
            p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])
            p_x = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()])
            p_x = p_x.reshape((state_below.shape[0], k))

            pos = k * p_n[Y] / (p_w + k * p_n[Y]) * T.log(p_w)
            neg = (p_x / (p_x + k * p_n[noise].reshape(p_x.shape)) * T.log(p_x)).sum(axis=1)


        return -(pos - neg).mean()


    def cost_(self, Y, Y_hat):
        # TODO fix me later when using IndexSpace

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
        state_below, = owner.inputs
        assert state_below.ndim == 2

        # TODO make this more generic like above
        state_below = state_below.owner.inputs[0].owner.inputs[0]

        #import ipdb
        #ipdb.set_trace()
        Y = T.argmax(Y, axis = 1)
        #Y = Y.astype('uint32')
        theano_rng = RandomStreams(seed = self.mlp.rng.randint(2 ** 15))
        noise = theano_rng.random_integers(size = (state_below.shape[0], self.num_noise_samples,), low=0, high = self.n_classes - 1)
        k = self.num_noise_samples
        p_n = 1. / self.n_classes

        pos = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y] - T.log(k * p_n))
        neg = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()] - T.log(k * p_n))
        # TODO is this reshape necessary?
        neg = neg.reshape((state_below.shape[0], k)).sum(axis=1)


        rval =  -T.log(pos) - T.log(1 - neg)
        return rval.mean()



    def nll(self, Y, Y_hat):
        """
        Expensive, use for monitoring
        """

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

        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        # we use sum and not mean because this is really one variable per row
        #Y = OneHotFormatter(self.n_classes).theano_expr(
                            #T.addbroadcast(Y, 1).dimshuffle(0).astype('uint32'))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1

        rval = log_prob_of.mean()

        return - rval


    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval = OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())])


        if target is not None:
            rval['nce'] = self.cost(Y_hat=state, Y=target)
            # NOTE expensive
            if not self.disable_ppl_monitor:
                rval['nll'] = self.nll(Y_hat=state, Y=target)
                rval['perplexity'] = 10 ** (rval['nll'] / np.log(10)).astype(config.floatX)
                rval['entropy'] = rval['nll'] / np.log(2).astype(config.floatX)

        return rval


class NCE2(Softmax):
    def __init__(self,
                num_noise_samples = 2,
                noise_prob = None,
                disable_ppl_monitor = True,
                **kwargs):

        super(NCE, self).__init__(**kwargs)
        self.num_noise_samples = num_noise_samples
        if noise_prob is not None:
            noise_prob = sharedX(noise_prob)
        self.noise_prob = noise_prob
        self.disable_ppl_monitor = disable_ppl_monitor
        #self.output_space = VectorSpace(1)


    def cost(self, Y, Y_hat):
        raise NotImplementedError()

    def get_monitoring_channels_from_state(self, state, target=None):

        mx = state.max(axis=1)

        rval = OrderedDict([('mean_max_class', mx.mean()),
                            ('max_max_class', mx.max()),
                            ('min_max_class', mx.min())])


        if target is not None:
            rval['nce'] = self.cost(Y_hat=state, Y=target)
            # NOTE expensive
            if not self.disable_ppl_monitor:
                rval['nll'] = self.nll(Y_hat=state, Y=target)
                rval['perplexity'] = 10 ** (rval['nll'] / np.log(10)).astype(config.floatX)
                rval['entropy'] = rval['nll'] / np.log(2).astype(config.floatX)

        return rval

    def score(self, Y, Y_hat):
        # TODO fix me later when using IndexSpace

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
        state_below, = owner.inputs
        assert state_below.ndim == 2

        # TODO make this more generic like above
        state_below = state_below.owner.inputs[0].owner.inputs[0]

        Y = T.argmax(Y, axis = 1)
        k = self.num_noise_samples

        if self.noise_prob is None:
            theano_rng = RandomStreams(seed = self.mlp.rng.randint(2 ** 15))
            noise = theano_rng.random_integers(size = (state_below.shape[0], self.num_noise_samples,), low=0, high = self.n_classes - 1)
            p_n = 1. / self.n_classes
            p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])
            p_x = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()])
            # TODO is this reshape necessary?
            p_x = p_x.reshape((state_below.shape[0], k))

            #pos = k * p_n / (p_w + k * p_n) * T.log(p_w)
            #neg = (p_x / (p_x + k * p_n) * T.log(p_x)).sum(axis=1)
        else:
            #import ipdb
            #ipdb.set_trace()
            theano_rng = MRG_RandomStreams(max(self.mlp.rng.randint(2 ** 15), 1))
            assert self.mlp.batch_size is not None
            noise = theano_rng.multinomial(pvals = np.tile(self.noise_prob.get_value(), k * self.mlp.batch_size))
            noise = T.argmax(noise, axis = 1)
            p_n = self.noise_prob
            p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])
            p_x = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()])
            p_x = p_x.reshape((state_below.shape[0], k))

            pos = k * p_n[Y] / (p_w + k * p_n[Y]) * T.log(p_w)
            neg = (p_x / (p_x + k * p_n[noise].reshape(p_x.shape)) * T.log(p_x)).sum(axis=1)


        #return -(pos - neg).mean()
        return p_w, p_x


class vLBL(Model):

    def __init__(self, dict_size, dim, context_length, k, irange = 0.1,
                 seed = 22, max_col_norm = None, min_col_norm = None,
                 y_max_col_norm = None, y_min_col_norm = None,
                 c_max_col_norm = None, c_min_col_norm = None):

        self.__dict__.update(locals())
        del self.self
        rng = np.random.RandomState(seed)
        self.rng = rng
        C = rng.uniform(-irange, irange, (dim, context_length))
        self.C = sharedX(C)

        W = rng.uniform(-irange, irange, (dict_size, dim))
        W = sharedX(W)
        self.projector = MatrixMul(W)
        W = rng.uniform(-irange, irange, (dict_size, dim))
        W = sharedX(W)
        self.y_projector = MatrixMul(W)
        self.b = sharedX(np.zeros((dict_size,)), name = 'vLBL_b')
        self.set_spaces()
        self.rng =  np.random.RandomState(2014)

    def set_spaces(self):
        self.input_space = IndexSpace(dim=self.context_length, max_labels=self.dict_size)
        self.output_space = VectorSpace(dim=self.dict_size)

    def get_params(self):

        rval = self.projector.get_params()
        rval.extend(self.y_projector.get_params())
        rval.extend([self.C, self.b])
        return rval

    def _modify_updates(self, updates):

        if self.max_col_norm is not None:
            W, = self.projector.get_params()
            if self.min_col_norm is None:
                self.min_col_norm = 0.
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms,
                                       self.min_col_norm,
                                       self.max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)

        if self.y_max_col_norm is not None:
            W, = self.y_projector.get_params()
            if self.y_min_col_norm is None:
                self.y_min_col_norm = 0.
            if W in updates:
                updated_W = updates[W]
                col_norms = T.sqrt(T.sum(T.sqr(updated_W), axis=0))
                desired_norms = T.clip(col_norms,
                                       self.y_min_col_norm,
                                       self.y_max_col_norm)
                updates[W] = updated_W * desired_norms / (1e-7 + col_norms)



        if self.c_max_col_norm is not None:
            if self.c_min_col_norm is None:
                self.c_min_col_norm = 0.
            if self.C in updates:
                updated_C = updates[self.C]
                col_norms = T.sqrt(T.sum(T.sqr(updated_C), axis=0))
                desired_norms = T.clip(col_norms,
                                       self.c_min_col_norm,
                                       self.c_max_col_norm)
                updates[self.C] = updated_C * desired_norms / (1e-7 + col_norms)

    def context(self, state_below):
        "q^(h) from EQ. 2"

        state_below = state_below.reshape((state_below.shape[0], self.dim, self.context_length))
        rval = self.C.dimshuffle('x', 0, 1) * state_below
        rval = rval.sum(axis=2)

        return rval

    def score(self, X, Y=None, k = 1):
        X = self.projector.project(X)
        q_h = self.context(X)
        # this is used during training
        if Y is not None:
            if Y.ndim != 1:
                Y = Y.flatten().dimshuffle(0)
            q_w = self.y_projector.project(Y)
            rval = (q_w.reshape((k, X.shape[0], q_h.shape[1])) * q_h).sum(axis=2)
            rval = rval + self.b[Y].reshape((k, X.shape[0]))
        # during nll
        else:
            q_w = self.y_projector._W
            rval = T.dot(q_h, q_w.T) + self.b.dimshuffle('x', 0)

        return rval

    def cost_from_X(self, data):
        X, Y = data
        z = self.score(X)
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1
        rval = as_floatX(log_prob_of.mean())
        return - rval

    def get_monitoring_data_specs(self):

        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)


    def nll(self, data):

        return self.cost_from_X(data)

    def get_monitoring_channels(self, data):
        X, Y = data
        rval = OrderedDict()

        nll = self.nll(data)
        rval['perplexity'] = as_floatX(10 ** (nll/np.log(10)))

        W, = self.projector.get_params()
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        rval['projector_row_norms_min'] = row_norms.min()
        rval['projector_row_norms_mean'] = row_norms.mean()
        rval['projector_row_norms_max'] = row_norms.max()
        rval['projector_col_norms_min'] = col_norms.min()
        rval['projector_col_norms_mean'] = col_norms.mean()
        rval['projector_col_norms_max'] = col_norms.max()

        W, = self.y_projector.get_params()
        sq_W = T.sqr(W)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        rval['y_projector_row_norms_min'] = row_norms.min()
        rval['y_projector_row_norms_mean'] = row_norms.mean()
        rval['y_projector_row_norms_max'] = row_norms.max()
        rval['y_projector_col_norms_min'] = col_norms.min()
        rval['y_projector_col_norms_mean'] = col_norms.mean()
        rval['y_projector_col_norms_max'] = col_norms.max()




        sq_W = T.sqr(self.C)
        row_norms = T.sqrt(sq_W.sum(axis=1))
        col_norms = T.sqrt(sq_W.sum(axis=0))
        rval['C_row_norms_min'] = row_norms.min()
        rval['C_row_norms_mean'] = row_norms.mean()
        rval['C_row_norms_max'] = row_norms.max()
        rval['C_col_norms_min'] = col_norms.min()
        rval['C_col_norms_mean'] = col_norms.mean()
        rval['C_col_norms_max'] = col_norms.max()

        return rval


class vLBL_NCE(vLBL):

    def __init__(self, batch_size, noise_p = None, **kwargs):

        super(vLBL_NCE, self).__init__(**kwargs)
        self.__dict__.update(locals())
        del self.self

        self.noise = sharedX(np.zeros(batch_size * self.k), dtype = 'int64')

        if noise_p is None:
            self.uniform = True
            self.noise_p = sharedX(1. / dict_size)
        else:
            self.uniform = False
            self.noise_p = sharedX(noise_p)
            self.theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

    def set_spaces(self):
        self.input_space = IndexSpace(dim=self.context_length, max_labels=self.dict_size)
        self.output_space = IndexSpace(dim=1, max_labels=self.dict_size)


    def delta(self, data, k = 1):

        X, Y = data
        if Y.ndim != 1:
            Y = Y.flatten().dimshuffle(0)

        if self.uniform:
            rval = self.score(X, Y) - T.log(self.k * self.noise_p)
        else:
            rval = self.score(X, Y, k = k)
            rval = rval - T.log(self.k * self.noise_p[Y]).reshape(rval.shape)
        return T.cast(rval, config.floatX)


    def get_noise(self):

        if self.uniform:
            if self.batch_size is None:
                raise NameError("Since numpy random is faster, batch_size is required")
            #noise = self.rng.randint(0, self.dict_size - 1, self.batch_size * self.k)
            #self.noise.set_value(noise)
            rval = self.theano_rng.random_integers(low = 0, high = self.dict_size -1,
                                                    size = (self.batch_size * self.k))
        else:
            #rval = self.rng.multinomial(n = 1, pvals = self.noise_p.get_value(), size = self.batch_size * self.k)
            rval = self.theano_rng.multinomial(pvals = T.repeat(self.noise_p.reshape((1, self.dict_size)),repeats= self.k * self.batch_size, axis=0))
            rval = T.argmax(rval, axis=1)
        return rval

    def cost_from_X(self, data):
        X, Y = data

        pos = T.nnet.sigmoid(self.delta(data))
        neg = 1. - T.nnet.sigmoid((self.delta((X, self.get_noise()), self.k)))
        neg = neg.sum(axis=0)

        rval = T.log(pos) + self.k * T.log(neg)
        #rval = T.log(pos) + T.log(neg)
        return -rval.mean()


    def nll(self, data):
        X, Y = data
        z = self.score(X)
        z = z - z.max(axis=1).dimshuffle(0, 'x')
        log_prob = z - T.log(T.exp(z).sum(axis=1).dimshuffle(0, 'x'))
        Y = OneHotFormatter(self.dict_size).theano_expr(Y)
        Y = Y.reshape((Y.shape[0], Y.shape[2]))
        log_prob_of = (Y * log_prob).sum(axis=1)
        assert log_prob_of.ndim == 1
        rval = as_floatX(log_prob_of.mean())
        return - rval


    def get_monitoring_channels(self, data):

        rval = super(vLBL_NCE, self).get_monitoring_channels(data)
        X, Y = data

        pos = T.nnet.sigmoid(self.delta(data))
        neg = 1. - T.nnet.sigmoid((self.delta((X, self.get_noise()), self.k)))
        neg = neg.sum(axis=0)

        rval['pos'] = -T.log(pos).mean()
        rval['neg'] = -T.log(neg).mean()
        rval['noise'] = T.cast(self.get_noise().sum(), config.floatX)
        return rval

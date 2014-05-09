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
from pylearn2.space import VectorSpace
from pylearn2.format.target_format import OneHotFormatter
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.utils import sharedX
#import ipdb


class NCE(Softmax):

    def __init__(self, num_noise_samples = 2, noise_prob = None,  **kwargs):

        super(NCE, self).__init__(**kwargs)
        self.num_noise_samples = num_noise_samples
        if noise_prob is not None:
            noise_prob = sharedX(noise_prob)
        self.noise_prob = noise_prob
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
            #y_hat = T.argmax(state, axis=1)
            #y = T.argmax(target, axis=1)
            #misclass = T.neq(y, y_hat).mean()
            #misclass = T.cast(misclass, config.floatX)
            #rval['misclass'] = misclass
            rval['nce'] = self.cost(Y_hat=state, Y=target)
            # NOTE expensive
            rval['nll'] = self.nll(Y_hat=state, Y=target)
            rval['perplexity'] = 10 ** (rval['nll'] / np.log(10)).astype(config.floatX)
            rval['entropy'] = rval['nll'] / np.log(2).astype(config.floatX)

        return rval


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
#import ipdb


class MLP(MLP):

    def cost_from_X(self, data):
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        state_below = X
        state_below = self.layers[0].fprop(state_below)
        for layer in self.layers[1:-1]:
            state_below = layer.fprop(state_below)
        return self.layers[-1].cost(state_below, Y)

    def get_monitoring_channels(self, data):

        X, Y = data
        state = X
        rval = OrderedDict()

        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + ".get_monitoring_channels did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                doc = 'This channel came from a layer called "' + \
                        layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name+'_'+key] = value
            old_state = state
            state = layer.fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
                state = old_state
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + \
                            ".get_monitoring_channels_from_state did" + \
                            " not provide any further documentation for" + \
                            " this channel."
                doc = 'This channel came from a layer called "' + \
                        layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name+'_'+key] = value

        return rval



class NCE(Softmax):


    def __init__(self, num_noise_samples = 2, **kwargs):

        super(NCE, self).__init__(**kwargs)
        self.num_noise_samples = num_noise_samples
        self.output_space = VectorSpace(1)


    #def fprop(self, state_below):
        ## TODO this is a hack to fix the monitorinf for now
        #return state_below

    #def fprop(self, state_below):
#
        #p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])




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
        theano_rng = RandomStreams(seed = self.mlp.rng.randint(2 ** 15))
        noise = theano_rng.random_integers(size = (state_below.shape[0], self.num_noise_samples,), low=0, high = self.n_classes - 1)
        k = self.num_noise_samples
        p_n = 1. / self.n_classes
        p_w = T.nnet.sigmoid((state_below * self.W[:, Y].T).sum(axis=1) + self.b[Y])
        p_x = T.nnet.sigmoid((T.concatenate([state_below] * k) * self.W[:, noise.flatten()].T).sum(axis=1) + self.b[noise.flatten()])
        # TODO is this reshape necessary?
        p_x = p_x.reshape((state_below.shape[0], k))

        pos = k * p_n / (p_w + k * p_n) * T.log(p_w)
        neg = (p_x / (p_x + k * p_n) * T.log(p_x)).sum(axis=1)


        #pos = T.nnet.sigmoid(T.dot(state_below, self.W[:,Y]) + self.b[Y] - T.log(self.num_noise_samples * p_n))
        #neg = T.nnet.sigmoid(T.dot(state_below, self.W[:,noise].T) + self.b[noise] - T.log(k * p_n))

        return - (pos - neg).mean()


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
        Y = OneHotFormatter(self.n_classes).theano_expr(
                            T.addbroadcast(Y, 1).dimshuffle(0).astype('uint32'))
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


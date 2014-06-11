from itertools import izip
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from theano.tensor.shared_randomstreams import RandomStreams


class NCE(DefaultDataSpecsMixin, Cost):

    supervised = True

    def expr( self, model, data, ** kwargs):
        return None

    def get_gradients(self, model, data, ** kwargs):

        space,  sources = self.get_data_specs(model)
        space.validate(data)
        X, Y = data


        theano_rng = RandomStreams(seed = model.rng.randint(2 ** 15))
        noise = theano_rng.random_integers(size = (X.shape[0], model.k,), low=0, high = model.dict_size - 1)


        delta = model.delta(data)
        p = model.score(X, Y)
        params = model.get_params()

        pos = T.jacobian(model.score(X, Y), params, disconnected_inputs='ignore')
        import ipdb
        ipdb.set_trace()
        pos = [(1 - T.nnet.sigmoid(model.delta(data))) * item for item in pos]
        #neg = 0
        #for i in xrange(model.k):
            #neg += T.nnet.sigmoid(model.delta((X, noise[:,i]))) * T.jacobian(model.score(X, noise[:, i]), params, disconnected_inputs='ignore')

        #grads = pos - neg
        grads = pos
        grads = gras.mean(axis=0)
        gradients = OrderedDict(izip(params, grads))
        updates = OrderedDict()

        return gradients, updates


class NCE_MLP(DefaultDataSpecsMixin, Cost):

    def expr(self, model, data, ** kwargs):
        return None

    def get_gradients(self, model, data, ** kwargs):

        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y ) = data
        Y_hat = model.layers[0].fprop(X)
        for i in range(1, len(model.layers) -1):
            Y_hat = model.layers[i].fprop(Y_hat)

        p_w, p_x = model.layers[-1](Y, Y_hat)


        #n_classes = model.layers[-1].n_classes
        #num_noise_samples = model.layers[-1].num_noise_samples


        #theano_rng = RandomStream(seed = model.rng.randint(2 ** 15))
        #noise = theano_rng.random_integers(size = (state_below.shape[0], num_noise_samples,), low=0, high = n_classes - 1)
#
        #p_n = 1. / n_classes
        #p_w = T.nnet.sigmoid((Y-hat, * self.W[:, Y].T)).sum(axis=1) +

        pos = (1 - T.sigmoid(delta)) * T.grad(P, params, disconnected_inputs='ignore')
        neg = ()
        grads = pos - neg
        gradients = OrderedDict(izip(params, grads))
        updates = OrderedDict()

        return  gradients, updates


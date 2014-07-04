from itertools import izip
import theano.tensor as T
from theano.compat.python2x import OrderedDict
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.costs.cost import DefaultDataSpecsMixin, Cost
from pylearn2.costs.mlp import Default as MLPDefaultCost
from pylearn2.space import CompositeSpace


class NCE(DefaultDataSpecsMixin, Cost):

    supervised = True

    def expr( self, model, data, ** kwargs):
        return None

    def get_gradients(self, model, data, ** kwargs):

        space,  sources = self.get_data_specs(model)
        space.validate(data)
        X, Y = data


        theano_rng = RandomStreams(seed = model.rng.randint(2 ** 15))
        noise = theano_rng.random_integers(size = (X.shape[0] * model.k,), low=0, high = model.dict_size - 1)


        delta = model.delta(data)
        p = model.score(X, Y)
        params = model.get_params()

        pos_ = T.jacobian(model.score(X, Y), params, disconnected_inputs='ignore')
        pos_coeff = 1 - T.nnet.sigmoid(model.delta(data))
        pos = []
        for param in pos_:
            axes = [0]
            axes.extend(['x' for item in range(param.ndim - 1)])
            pos.append(pos_coeff.dimshuffle(axes) * param)
        del pos_, pos_coeff

        noise_x = T.tile(X, (model.k, 1))
        neg_ = T.jacobian(model.score(noise_x, noise), params, disconnected_inputs='ignore')
        neg_coeff = T.nnet.sigmoid(model.delta((noise_x, noise)))
        neg = []
        for param in neg_:
            axes = [0]
            axes.extend(['x' for item in range(param.ndim - 1)])
            tmp = neg_coeff.dimshuffle(axes) * param
            new_shape = [X.shape[0], model.k]
            new_shape.extend([tmp.shape[i] for i in range(1, tmp.ndim)])
            neg.append(tmp.reshape(new_shape).sum(axis=1))
        del neg_, neg_coeff


        grads = [(pos_ - neg_).mean(axis=0) for pos_, neg_ in zip(pos, neg)]
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


class NCE_Noise(MLPDefaultCost):

    def get_data_specs(self, model):

        spaces = CompositeSpace([model.get_input_space(),
                                model.get_output_space(),
                                model.get_noise_space()])
        sources = (model.get_input_source(), model.get_target_source(), model.get_noise_source())
        return (spaces, sources)


class StraightThrough(MLPDefaultCost):

    def get_gradients(self, model, data, ** kwargs):
        """
        Provides the gradients of the cost function with respect to the model
        parameters.

        These are not necessarily those obtained by theano.tensor.grad
        --you may wish to use approximate or even intentionally incorrect
        gradients in some cases.

        Parameters
        ----------
        model : a pylearn2 Model instance
        data : a batch in cost.get_data_specs() form
        kwargs : dict
            Optional extra arguments, not used by the base class.

        Returns
        -------
        gradients : OrderedDict
            a dictionary mapping from the model's parameters
            to their gradients
            The default implementation is to compute the gradients
            using T.grad applied to the value returned by expr.
            However, subclasses may return other values for the gradient.
            For example, an intractable cost may return a sampling-based
            approximation to its gradient.
        updates : OrderedDict
            a dictionary mapping shared variables to updates that must
            be applied to them each time these gradients are computed.
            This is to facilitate computation of sampling-based approximate
            gradients.
            The parameters should never appear in the updates dictionary.
            This would imply that computing their gradient changes
            their value, thus making the gradient value outdated.
        """

        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError, e:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            e.message += " while calling " + str(type(self)) + ".expr"
            logger.error(type(self))
            logger.error(e.message)
            raise e

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        # we make the assumption that last layer grad is computable
        params = model.layers[-1].get_params()
        grads = T.grad(cost, params, disconnected_inputs='ignore')

        for layer in model.layers[:-1][::-1]:
            p = layer.get_params()
            if hasattr(layer, "get_gradients"):
                #g = layer.get_gradients(grads[-1])
                g = layer.get_gradients(cost)
            else:
                g = T.grad(cost, p, disconnected_inputs='ignore')

            grads.extend(g)
            params.extend(p)


        #params = list(model.get_params())

        #grads = T.grad(cost, params, disconnected_inputs='ignore')

        gradients = OrderedDict(izip(params, grads))

        updates = OrderedDict()

        return gradients, updates


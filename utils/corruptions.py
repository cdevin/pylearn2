import theano
from theano import tensor
from pylearn2.corruption import Corruptor

class BinomialCorruptorScaled(Corruptor):
    """
    A binomial corruptor sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """
    def _corrupt(self, x):
        scale = self.corruption_level if self.corruption_level != 0 else 1.
        return self.s_rng.binomial(
            size=x.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * x / scale

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a binomial (masking) noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted inputs,
            where individual inputs have been masked with independent
            probability equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        else:
            return [self._corrupt(inp) for inp in inputs]

class BinomialCorruptorScaledGroup(Corruptor):
    """
    A binomial corruptor sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """
    def __init__(self, corruption_level, group_size, rng=2001):

        self.group_size = group_size
        super(BinomialCorruptorScaledGroup, self).__init__(
                corruption_level,
                rng)

    def _corrupt(self, x):
        scale = self.corruption_level if self.corruption_level != 0 else 1.
        noise = self.s_rng.binomial(
            size=(x.shape[0], self.group_size),
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX)

        noise = tensor.extra_ops.repeat(noise, x.shape[1] / self.group_size, axis =1)
        return noise * x / scale

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with a binomial (masking) noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like
            Theano symbolic representing the corresponding corrupted inputs,
            where individual inputs have been masked with independent
            probability equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        else:
            return [self._corrupt(inp) for inp in inputs]

class GaussianCorruptor(Corruptor):
    """
    A Gaussian corruptor transforms inputs by adding zero
    mean isotropic Gaussian noise.
    """

    def __init__(self, stdev, avg = 0., rng=2001):
        self.avg = avg
        super(GaussianCorruptor, self).__init__(corruption_level=stdev, rng=rng)

    def _corrupt(self, x):
        noise = self.s_rng.normal(
            size=x.shape,
            avg=self.avg,
            std=self.corruption_level,
            dtype=theano.config.floatX
        )

        rval = noise + x

        return rval

    def __call__(self, inputs):
        """
        (Symbolically) corrupt the inputs with Gaussian noise.

        Parameters
        ----------
        inputs : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing a (list of) (mini)batch of inputs
            to be corrupted, with the first dimension indexing training
            examples and the second indexing data dimensions.

        Returns
        -------
        corrupted : tensor_like, or list of tensor_likes
            Theano symbolic(s) representing the corresponding corrupted inputs,
            where individual inputs have been corrupted by zero mean Gaussian
            noise with standard deviation equal to `self.corruption_level`.
        """
        if isinstance(inputs, tensor.Variable):
            return self._corrupt(inputs)
        return [self._corrupt(inp) for inp in inputs]

    def corruption_free_energy(self, corrupted_X, X):
        axis = range(1, len(X.type.broadcastable))

        rval = (T.sum(T.sqr(corrupted_X - X), axis=axis) /
                (2. * (self.corruption_level ** 2.)))
        assert len(rval.type.broadcastable) == 1
        return rval



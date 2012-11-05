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


class BinomialCorruptorScaledGroupCombined(Corruptor):
    """
    A binomial corruptor sets inputs to 0 with probability
    0 < `corruption_level` < 1.
    """
    def __init__(self, corruption_level_group, corruption_level_individual,
                group_size, rng=2001):

        self.group_size = group_size
        super(BinomialCorruptorScaledGroupCombined, self).__init__(
                corruption_level_individual, rng)
        self.corruption_level_group = corruption_level_group
        self.corruption_level_individual = corruption_level_individual

    def _corrupt(self, x):
        scale_group = self.corruption_level_group if self.corruption_level_group != 0 else 1.
        noise_g = self.s_rng.binomial(
            size=(x.shape[0], self.group_size),
            n=1,
            p=1 - self.corruption_level_group,
            dtype=theano.config.floatX)
        noise_g = tensor.extra_ops.repeat(noise_g, x.shape[1] / self.group_size, axis =1)

        scale_individual = self.corruption_level_individual if self.corruption_level_individual != 0 else 1.
        noise_i = self.s_rng.binomial(
            size=x.shape,
            n=1,
            p=1 - self.corruption_level_individual,
            dtype=theano.config.floatX)


        return noise_g * noise_i * x / (scale_group * scale_individual)

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



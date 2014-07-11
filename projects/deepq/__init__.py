from pylearn2.models.mlp import Softmax
from pylearn2.models.mlp import MLP
from pylearn2.utils import block_gradient, constantX
from theano import tensor as T
from theano.printing import Print
from  theano.printing import debugprint

class GaterSoftmax(Softmax):


    def __init__(self,  **kwargs):
        # num_classes is acutally num_experts and vice versa
        super(GaterSoftmax, self).__init__(**kwargs)
        self.output_space.dim = 10


    def fprop(self, state_below):


        z = super(GaterSoftmax, self).fprop(state_below)

        assert hasattr(z, 'owner')
        owner = z.owner
        assert owner is not None
        op = owner.op
        if isinstance(op, Print):
            assert len(owner.inputs) == 1
            z, = owner.inputs
            owner = z.owner
            op = owner.op
        assert isinstance(op, T.nnet.Softmax)
        z, = owner.inputs
        assert z.ndim == 2

        experts = block_gradient(self.mlp.experts_fprop)
        z = (z.dimshuffle(1, 0, 'x') * experts).sum(axis=0)

        rval = T.nnet.softmax(z)

        return rval


class GaterMLP(MLP):

    def __init__(self, experts, **kwargs):
        super(GaterMLP, self).__init__(**kwargs)
        self.experts = experts


    def fprop_(self, state_below, return_all=False):

        rval = self.layers[0].fprop(state_below)

        rlist = [rval]

        for layer in self.layers[1:]:

            if isinstance(layer, GaterSoftmax):
                tmp = [expert.fprop(state_below) for expert in self.experts]
                tmp = T.concatenate([item.dimshuffle('x', 0, 1) for item in tmp], axis=0)
                rval = layer.fprop(rval, tmp)
            else:
                rval = layer.fprop(rval)
                rlist.append(rval)

        if return_all:
            return rlist
        return rval


    def fprop(self, state_below, return_all=False):

        tmp = [expert.fprop(state_below) for expert in self.experts]
        self.experts_fprop = T.concatenate([item.dimshuffle('x', 0, 1) for item in tmp], axis=0)
        return super(GaterMLP, self).fprop(state_below, return_all)

    def dropout_fprop(self, state_below, default_input_include_prob=0.5,
        input_include_probs=None, default_input_scale=2.,
        input_scales=None, per_example=True):

        tmp = [expert.fprop(state_below) for expert in self.experts]
        self.experts_fprop = T.concatenate([item.dimshuffle('x', 0, 1) for item in tmp], axis=0)
        return super(GaterMLP, self).dropout_fprop(state_below, default_input_include_prob,
                                            input_include_probs, default_input_scale)


    def get_monitoring_channels(self, data):

        X, y = data
        tmp = [expert.fprop(X) for expert in self.experts]
        self.experts_fprop = T.concatenate([item.dimshuffle('x', 0, 1) for item in tmp], axis=0)
        return super(GaterMLP, self).get_monitoring_channels(data)

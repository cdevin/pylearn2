from pylearn2.models.mlp import Softmax
from pylearn2.space import VectorSpace
from theano import config
from theano import tensor as T
from theano.tensor.xlogx import xlogx
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams


INF = 1000000
INF = 0

class TreeSoftmax(Softmax):

    def __init__(self, n_classes, layer_name, irange = None,
            istdev = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False,
                 max_col_norm = None, init_bias_target_marginals= None,
                 stochastic = True):


        super(TreeSoftmax, self).__init__(n_classes = n_classes,
                                            layer_name = layer_name,
                                            irange = irange,
                                            istdev = istdev,
                                            sparse_init = sparse_init,
                                            W_lr_scale = W_lr_scale,
                                            b_lr_scale = b_lr_scale,
                                            max_row_norm = max_row_norm,
                                            no_affine = no_affine,
                                            max_col_norm = max_col_norm,
                                            init_bias_target_marginals = init_bias_target_marginals)
        self.output_space = VectorSpace(10)
        self.theano_rng = MRG_RandomStreams(2 ** 15)
        self.stochastic = stochastic

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary.
        returns: \sum_s p(s)logp(s) - \sum_s \sum_c p(c|s)p(s)logp(c|s)p(s)
        """


        #p(s) log p(s)
        ps = Y_hat.sum(axis=0)/Y_hat.shape[0]
        pslog = xlogx(ps)
        pslog = T.switch(T.isnan(pslog), 0, pslog)
        pslog = T.switch(T.isinf(pslog), INF, pslog)

        # this works only in binary case
        if not self.stochastic:
            Y_hat_ = T.argmax(Y_hat_, axis=1, keepdims=True)
            pc0 = (Y * Y_hat_).sum(axis=0) * ps[0]  # p(c|0) * p(s=0)
            pc1 = (Y * T.neq(Y_hat_, 1)).sum(axis=0) * ps[1]  # p(c|1) * p(s=1)
        else:
            Y_hat_ = self.theano_rng.multinomial(pvals = Y_hat)
            pc0 = (Y * Y_hat_[:,0].reshape((Y_hat_.shape[0], 1))).sum(axis=0) * ps[0]  # p(c|0) * p(s=0)
            pc1 = (Y * Y_hat_[:,1].reshape((Y_hat_.shape[0], 1))).sum(axis=0) * ps[1]  # p(c|0) * p(s=0)
        pc0 = pc0 / Y_hat.shape[0]
        pc1 = pc1 / Y_hat.shape[0]

        # \sum_s p(c,s) * log p(c,s)
        #pcs = xlogx(pc0) + xlogx(pc1)
        pcs = pc0 * T.log(pc0) + pc1 * T.log(pc1)
        pcs = T.switch(T.isnan(pcs), 0, pcs)
        pcs = T.switch(T.isinf(pcs), INF, pcs)

        cost = pslog.sum() - pcs.sum()

        #import ipdb
        #ipdb.set_trace()

        return cost.astype(config.floatX)


    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict([])

        Y = target
        Y_hat = state
        rval['misclass'] = 0.


        ps = Y_hat.sum(axis=0)/Y_hat.shape[0]
        Y_hat_ = T.argmax(Y_hat, axis=1, keepdims=True).astype(config.floatX)
        rval['ps_l'] = ps[0].astype(config.floatX)
        rval['ps_r'] = ps[1].astype(config.floatX)
        rval['y_hat_std'] = Y_hat_.std().astype(config.floatX)

        return rval



from pylearn2.models.mlp import Softmax, Sigmoid
from pylearn2.models.mlp import Layer
from pylearn2.space import VectorSpace
from theano import config
from theano import tensor as T
from theano.tensor.xlogx import xlogx
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams


INF = 1000000
INF = 0


class TreeSigmoid(Sigmoid):

    def __init__(self, **kwargs):

        super(TreeSigmoid, self).__init__(**kwargs)

    def set_input_space(self, space):
        super(TreeSigmoid, self).set_input_space(space)
        self.output_space = VectorSpace(10)


    def cost(self, Y, Y_hat):


        #ps = Y_hat.mean(axis=0)
        log_ps = T.log(Y_hat)
        log_ps = T.switch(T.isinf(log_ps), 0., log_ps)
        log_1_ps = T.log(1-Y_hat)
        log_1_ps = T.switch(T.isinf(log_1_ps), 0., log_1_ps)
        s_ps = Y_hat * log_ps + (1-Y_hat) * log_1_ps

        pcs = Y * T.addbroadcast(Y_hat,1)
        log_pcs = T.log(pcs)
        log_pcs = T.switch(T.isinf(log_pcs), 0., log_pcs)
        pcs_1 = Y * T.addbroadcast(1-Y_hat, 1)
        log_1_pcs = T.log(pcs_1)
        log_1_pcs = T.switch(T.isinf(log_1_pcs), 0., log_1_pcs)
        s_pcs = pcs * log_pcs + pcs_1 * log_1_pcs


        #ps = Y_hat.mean(axis=0)
        #log_ps = T.log(ps)
        #log_ps = T.switch(T.isinf(log_ps), 0., log_ps)
        #log_1_ps = T.log(1-ps)
        #log_1_ps = T.switch(T.isinf(log_1_ps), 0., log_1_ps)
        #s_ps = ps * log_ps + (1-ps) * log_1_ps

        #pcs = (Y_hat * Y).sum(axis=0)
        #log_pcs = T.log(pcs)
        #log_pcs = T.switch(T.isinf(log_pcs), 0., log_pcs)
        #log_1_pcs = T.log(1-pcs)
        #log_1_pcs = T.switch(T.isinf(log_1_pcs), 0., log_1_pcs)
        #s_pcs = pcs * log_pcs + (1-pcs) * log_1_pcs

        return s_ps.sum() - s_pcs.sum()



    #def cost(self, Y, Y_hat):
        #"""
        #Y must be one-hot binary.
        #returns: \sum_s p(s)logp(s) - \sum_s \sum_c p(c|s)p(s)logp(c|s)p(s)
        #"""

        #ps = Y_hat.mean()

        #ps1 = T.gt(Y_hat, 0.5)
        #ps1 = T.addbroadcast(ps1, 1)
        #ps0 = T.le(Y_hat, 0.5)
        #ps0 = T.addbroadcast(ps0, 1)
        #pcs1 = (Y * ps1).sum(axis=0) / Y_hat.shape[0]
        #pcs0 = (Y * ps0).sum(axis=0) / Y_hat.shape[0]
        #pcs1log = T.log(pcs1 * ps)
        #pcs1log = T.switch(T.isinf(pcs1log), 0, pcs1log)
        #pcs0log = T.log(pcs0 * (1-ps))
        #pcs0log = T.switch(T.isinf(pcs0log), 0, pcs0log)

        #pslog1 = T.log(ps)
        #pslog1 = T.switch(T.isinf(pslog1), 0, pslog1)
        #pslog1 = pslog1 * ps
        #pslog0 = T.log(1-ps)
        #pslog0 = T.switch(T.isinf(pslog0), 0, pslog0)
        #pslog0 = pslog0 * (1-ps)

        #cost = (pslog0 + pslog1).sum()
        #cost += (pcs0log + pcs1log).sum()

        #return - cost.sum().astype(config.floatX)

    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict([])

        Y = target
        Y_hat = state

        #ps = Y_hat.mean()
        right = T.gt(Y_hat, 0.5).sum()
        left = T.le(Y_hat, 0.5).sum()

        right_class = Y * T.addbroadcast(T.gt(Y_hat, 0.5),1)
        left_class = Y * T.addbroadcast(T.le(Y_hat, 0.5),1)
        #ps1 = T.gt(Y_hat, 0.5)
        #ps1 = T.addbroadcast(ps1, 1)
        #ps0 = T.le(Y_hat, 0.5)
        #ps0 = T.addbroadcast(ps0, 1)
        #pcs1 = (Y * ps1).sum(axis=0) / Y_hat.shape[0]
        #pcs0 = (Y * ps0).sum(axis=0) / Y_hat.shape[0]
        #pcs1log = T.log(pcs1)
        #pcs1log = T.switch(T.isinf(pcs1log), 0, pcs1log)
        #pcs0log = T.log(pcs0)
        #pcs0log = T.switch(T.isinf(pcs0log), 0, pcs0log)
        #cost_r = pcs1 * ps * pcs1log
        #cost_l = pcs0 * (1-ps) * pcs0log



        rval['right'] = right.astype(config.floatX)
        rval['left'] = left.astype(config.floatX)
        #rval['cost_r'] = cost_r.sum().astype(config.floatX)
        #rval['cost_l'] = cost_l.sum().astype(config.floatX)
        rval['ps'] = Y_hat.mean().astype(config.floatX)
        rval['ps_max'] = Y_hat.max().astype(config.floatX)
        rval['ps_min'] = Y_hat.min().astype(config.floatX)
        #rval['r_0'] = right_class[:,0].sum().astype(config.floatX)
        #rval['l_0'] = left_class[:,0].sum().astype(config.floatX)
        #rval['r_1'] = right_class[:,1].sum().astype(config.floatX)
        #rval['l_1'] = left_class[:,1].sum().astype(config.floatX)
        #rval['r_2'] = right_class[:,2].sum().astype(config.floatX)
        #rval['l_2'] = left_class[:,2].sum().astype(config.floatX)
        #rval['r_3'] = right_class[:,3].sum().astype(config.floatX)
        #rval['l_3'] = left_class[:,3].sum().astype(config.floatX)
        #rval['r_4'] = right_class[:,4].sum().astype(config.floatX)
        #rval['l_4'] = left_class[:,4].sum().astype(config.floatX)
        #rval['r_5'] = right_class[:,5].sum().astype(config.floatX)
        #rval['l_5'] = left_class[:,5].sum().astype(config.floatX)
        #rval['r_6'] = right_class[:,6].sum().astype(config.floatX)
        #rval['l_6'] = left_class[:,6].sum().astype(config.floatX)
        #rval['r_7'] = right_class[:,7].sum().astype(config.floatX)
        #rval['l_7'] = left_class[:,7].sum().astype(config.floatX)
        #rval['r_8'] = right_class[:,8].sum().astype(config.floatX)
        #rval['l_8'] = left_class[:,8].sum().astype(config.floatX)
        #rval['r_9'] = right_class[:,9].sum().astype(config.floatX)
        #rval['l_9'] = left_class[:,9].sum().astype(config.floatX)

        return rval

class TreeSoftmax(Softmax):

    def __init__(self, n_classes, layer_name, irange = None,
            istdev = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False,
                 max_col_norm = None, init_bias_target_marginals= None,
                 stochastic = False):


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
            Y_hat_ = T.argmax(Y_hat, axis=1, keepdims=True)
            pc0 = (Y * Y_hat_).sum(axis=0) * ps[0]  # p(c|0) * p(s=0)
            pc1 = (Y * T.neq(Y_hat_, 1)).sum(axis=0) * ps[1]  # p(c|1) * p(s=1)
        else:
            Y_hat_ = self.theano_rng.multinomial(pvals = Y_hat)
            pc0 = (Y * Y_hat_[:,0].reshape((Y_hat_.shape[0], 1))).sum(axis=0) * ps[0]
            pc1 = (Y * Y_hat_[:,1].reshape((Y_hat_.shape[0], 1))).sum(axis=0) * ps[1]

        pc0 = pc0 / Y_hat.shape[0]
        pc1 = pc1 / Y_hat.shape[0]

        # \sum_s p(c,s) * log p(c,s)
        #pcs = xlogx(pc0) + xlogx(pc1)
        pcs = pc0 * T.log(pc0) + pc1 * T.log(pc1)
        pcs = T.switch(T.isnan(pcs), 0, pcs)
        pcs = T.switch(T.isinf(pcs), INF, pcs)

        cost = pslog.sum() - pcs.sum()

        return cost.astype(config.floatX)


    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict([])

        Y = target
        Y_hat = state

        ps = Y_hat.sum(axis=0)/Y_hat.shape[0]
        Y_hat_ = T.argmax(Y_hat, axis=1, keepdims=True).astype(config.floatX)
        rval['ps_l'] = ps[0].astype(config.floatX)
        rval['ps_r'] = ps[1].astype(config.floatX)
        rval['y_hat_std'] = Y_hat_.std().astype(config.floatX)

        return rval


    #def prop(self, state_below):
        #state = super(TreeSoftmax, self).fprop(state_below)
        #return self.theano_rng.multinomial(pvals = state)

class PretrainedMLPLayer(Layer):
    """
    A layer whose weights are initialized, and optionally fixed,
    based on prior training.
    """

    def __init__(self, layer_name, layer_content, layer_num, freeze_params=False):
        """
        layer_content: A Model that implements "upward_pass", such as an
            RBM or an Autoencoder
        freeze_params: If True, regard layer_conent's parameters as fixed
            If False, they become parameters of this layer and can be
            fine-tuned to optimize the MLP's cost function.
        """
        self.__dict__.update(locals())
        self.layer_content = layer_content.layers[layer_num]
        del self.self

    def set_input_space(self, space):
        assert self.get_input_space() == space

    def get_params(self):
        if self.freeze_params:
            return []
        return self.layer_content.get_params()

    def get_input_space(self):
        return self.layer_content.get_input_space()

    def get_output_space(self):
        return self.layer_content.get_output_space()

    def fprop(self, state_below):
        return self.layer_content.fprop(state_below)




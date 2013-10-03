from pylearn2.models.mlp import Softmax
from pylearn2.space import VectorSpace
from theano import config
from theano import tensor as T
from theano.tensor.xlogx import xlogx
from theano.compat.python2x import OrderedDict

class TreeSoftmax(Softmax):

    def __init__(self, n_classes, layer_name, irange = None,
            istdev = None,
                 sparse_init = None, W_lr_scale = None,
                 b_lr_scale = None, max_row_norm = None,
                 no_affine = False,
                 max_col_norm = None, init_bias_target_marginals= None):


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

    def cost(self, Y, Y_hat):
        """
        Y must be one-hot binary. Y_hat is a softmax estimate.
        of Y. Returns negative log probability of Y under the Y_hat
        distribution.
        """

        #import ipdb
        #ipdb.set_trace()
        ps = xlogx(Y_hat.mean(axis=0)).sum()

        # this works only in binary case
        Y_hat_ = T.argmax(Y_hat, axis=1).reshape((Y_hat.shape[0], 1))
        pc0 = (Y * Y_hat_).mean(axis=0) * Y_hat[:,0].mean(axis=0)  # p(c|0)
        pc1 = (Y * T.neq(Y_hat_, 1)).mean(axis=0) * Y_hat[:,1].mean(axis=0)  # p(c|1)
        pcs = xlogx(pc0 + pc1).sum()

        return (ps - pcs).astype(config.floatX)


    def get_monitoring_channels_from_state(self, state, target=None):
        rval =  OrderedDict([])

        rval['misclass'] = 0.
        return rval



from pylearn2.models.mlp import MLP
from collections import OrderedDict
from theano import tensor
import theano
import ipdb

class MLP_IG(MLP):

    def get_grads(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        X, Y = data
        Y_hat = self.fprop(X)

        grads = OrderedDict()
        h = self.layers[0].fprop(X)
        for i in xrange(len(self.layers)):
            ig = self.information(h, Y)
            params = self.layers[i].get_params()
            for item in params:
                grads[item] = theano.gradient.jacobian(ig, item)
                #grad, _ = theano.scan(lambda i, ig, item : tensor.grad(ig[i], item[i,:]), sequences=tensor.arange(ig.shape[0]), non_sequences=[ig,item])
                #grads[item] = grad
                ipdb.set_trace()

            if i < len(self.layers) - 1:
                h = self.layers[i+1].fprop(h)


        return grads


    def information(self, h, y):

        mean_val = h.mean(axis=0)

        # divide data into two groups
        divid = tensor.ge(h, mean_val)

        # n_c
        n_class_per_div = divid.T.reshape((divid.shape[1], divid.shape[0], 1)) * y
        n_class_per_div = n_class_per_div.sum(axis=1)


        # n_c / n
        nc_n = n_class_per_div / y.sum(axis=0)

        # I
        I = nc_n * tensor.log(nc_n)
        I = I.sum(axis=1)

        return I



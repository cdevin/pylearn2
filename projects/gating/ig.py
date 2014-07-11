from pylearn2.models.mlp import RectifiedLinear, MLP
from collections import OrderedDict
from theano import tensor

class MyMLP(MLP):

    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = self.fprop(X, Y)
        return self.cost(Y, Y_hat)

    def fprop(self, state_below, Y, return_all = False):

        rval = self.layers[0].fprop(state_below)
        ig = self.get_ig(rval, Y)
        mask = tensor.ge(ig, ig.mean(()))

        rlist = [rval * mask]

        for layer in self.layers[1:]:
            rval = layer.fprop(rval)
            ig = self.get_ig(rval, Y)
            mask = tensor.ge(ig, ig.mean(()))
            rlist.append(rval * mask)

        if return_all:
            return rlist
        return rval



    def get_ig(self, h, y):

        mean_val = h.mean(axis=0)

        # divide data into two groups
        divid = tensor.ge(h, mean_val)

        # n_c
        n_class_per_div = divid.T.reshape((divid.shape[1], divid.shape[0], 1)) * y
        n_class_per_div = n_class_per_div.sum(axis=1)


        # n_c / n
        n = y.sum(axis=0)
        n = tensor.switch(tensor.eq(n, 0),-1, n)
        nc_n = n_class_per_div / n
        nc_n = tensor.ge(nc_n, 0) * nc_n

        # I
        I = nc_n * tensor.log(nc_n)
        I = tensor.switch(tensor.eq(nc_n,0), 0, I)
        I = I.sum(axis=1)

        return I

    def get_monitoring_channels(self, data):
        """
        data is a flat tuple, and can contain features, targets, or both
        """
        X, Y = data
        state = X
        rval = OrderedDict()

        for layer in self.layers:
            ch = layer.get_monitoring_channels()
            for key in ch:
                rval[layer.layer_name+'_'+key] = ch[key]
            state = layer.fprop(state)
            args = [state]
            if layer is self.layers[-1]:
                args.append(Y)
            ch = layer.get_monitoring_channels_from_state(*args)
            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                rval[layer.layer_name+'_'+key]  = ch[key]

            rval[layer.layer_name+'_ig'] = self.get_ig(state, Y).mean()

        return rval



from pylearn2.models.mlp import MLP
import ipdb

class MLP_IG(MLP):

    def train_batch(self, dataset, batch_size):
        """ A default learning rule based on SML """
        ipdb.set_trace()
        self.learn_mini_batch(dataset.get_batch_design(batch_size))
        return True

    def learn_mini_batch(self, X):
        """ A default learning rule based on SML """

        ipdb.set_trace()
        if not hasattr(self, 'learn_func'):
            self.redo_theano()

        rval =  self.learn_func(X)

        return rval

    def redo_theano(self):
        """ Compiles the theano function for the default learning rule """


        X = tensor.matrix()
        Y = tensor.matrix()
        update = get_updates(X, Y)
        self.learn_func = theano.function([minibatch], updates=updates)

    def cost_from_X_data_specs(self):
        """
        Returns the data specs needed by cost_from_X.

        This is useful if cost_from_X is used in a MethodCost.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_output_space()))
        source = (self.get_input_source(), self.get_target_source())
        return (space, source)



    #def cost(x, y):

        #rval = self.layers[0].fprop(x)
        #get ig

        #for item in mini batch:
            #rval[item_index] > thresehod:
                #y = +1
            #else:
                #y = -1
        #count(y+) count(y-)
        ##
        ##for layers in self


####
    def cost_from_X(self, data):
        """
        Computes self.cost, but takes data=(X, Y) rather than Y_hat as an argument.
        This is just a wrapper around self.cost that computes Y_hat by
        calling Y_hat = self.fprop(X)
        """
        self.cost_from_X_data_specs()[0].validate(data)
        X, Y = data
        Y_hat = self.fprop(X)


        for layer in self.layers:
            h = layer.fprop(X)
            ig = self.information(h, Y)

            self.update(ig, layer.W)

    def update(self, ig, w):

        theano.gradient.jacobian(ig, w)
        updates = {w: w - grad (* lr)}


    def information(self, h, y):

        ipdb.set_trace()
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

        ## num of items per class
        #y.sum(axis=0)

        ## num of classes
        #y.shape[1]

        return I

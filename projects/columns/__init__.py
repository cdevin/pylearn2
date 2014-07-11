import theano.tensor as T
#from theano.config import floatX
import theano
from pylearn2.models.mlp import Layer
from pylearn2.utils import serial
from pylearn2.space import VectorSpace, Conv2DSpace
import ipdb


class Columns(Layer):

    def __init__(self,
                layer_name,
                main_column,
                small_column,
                num_columns):

        self.main_column = main_column
        self.instantiate_columns(small_column, num_columns)
        self.layer_name = layer_name
        self.output_space = VectorSpace(self.columns[0].layers[-1].output_space.dim * num_columns)

    def instantiate_columns(self, column, num_columns):
        self.columns = []
        for i in xrange(num_columns):
            self.columns.append(serial.load_train_file(column))

    def get_params(self):
        params = []
        for i in xrange(len(self.columns)):
            params.extend(self.columns[i].get_params())
        return params

    def set_input_space(self, space):
        assert self.get_input_space() == space

    def get_input_space(self):
        return self.main_column.get_input_space()

    def fprop(self, state):

        # get the gaters
        z = state
        for i in xrange(len(self.main_column.layers) - 1):
            z = self.main_column.layers[i].fprop(z)

        z = T.switch(T.gt(z,0),1.,0.)

        col_z = []
        for i in xrange(len(self.columns)):
            col_z.append(self.columns[i].fprop(state) * z[:,i].dimshuffle(0, 'x'))

        z = T.concatenate(col_z, 1)

        return z

class ColumnsOneBatch(Columns):
    """ The case when mini-batch size is one.
        It will fail with batch size larger than one.
    """

    def fprop(self, state):

        z = state
        for i in xrange(len(self.main_column.layers) - 1):
            z = self.main_column.layers[i].fprop(z)

        z = T.switch(T.gt(z,0),1.,0.)
        zeros = T.alloc(0.0, 1, self.columns[i].get_output_space().dim).astype(theano.config.floatX)
        zeros.name = 'zeros place holder'
        col_z = []
        for i in xrange(len(self.columns)):
            col_z.append(T.switch(T.eq(z[0,i], 0), zeros, self.columns[i].fprop(state)))

        z = T.concatenate(col_z, 1).astype(theano.config.floatX)
        return z


class Columns_Connected(Layer):

    def __init__(self,
                layer_name,
                main_column,
                small_column,
                num_columns):

        self.main_column = main_column
        self.instantiate_columns(small_column, num_columns)
        self.layer_name = layer_name
        if isinstance(self.columns[0].layers[-1].output_space, VectorSpace):
            self.output_space = VectorSpace(self.columns[0].layers[-1].output_space.dim * num_columns)
        elif isinstance(self.columns[0].layers[-1].output_space, Conv2DSpace):
            m_c = self.main_column.layers[-2].output_space.num_channels +\
                    self.columns[0].layers[-1].output_space.num_channels
            sh = self.columns[0].layers[-1].output_space.shape
            self.output_space = Conv2DSpace(shape = sh, num_channels= m_c)

    def instantiate_columns(self, column, num_columns):
        self.columns = []
        for i in xrange(num_columns):
            self.columns.append(serial.load_train_file(column))
            # adjust the input_space
            #for j in xrange(1, len(self.columns[-1].layers)):
                #sp = self.columns[-1].layers[j].output_space

    def get_params(self):
        params = []
        for i in xrange(len(self.columns)):
            params.extend(self.columns[i].get_params())
        return params

    def set_input_space(self, space):
        assert self.get_input_space() == space

    def get_input_space(self):
        return self.main_column.get_input_space()

    def fprop(self, state):

        # get the gaters
        z = state
        z_list = []
        for i in xrange(len(self.main_column.layers) - 1):
            z = self.main_column.layers[i].fprop(z)
            z_list.append(z)

        gate = T.switch(T.gt(z,0),1.,0.)

        col_z = []
        for i in xrange(len(self.columns)):
            z_ = state
            for j in xrange(len(self.columns[i].layers)):
                z_ = self.columns[i].layers[j].fprop(z_)
                batch_index = self.main_column.layers[j].output_space.axes.index('c')
                z_ = T.concatenate([z_list[j], z_], batch_index)
            if z_.ndim == 2:
                #fully connected
                col_z.append(z_ * gate[:,i].dimshuffle(0, 'x'))
            else:
                col_z.append(z_ * gate[:,i])

        z = T.concatenate(col_z, 1)
        return z



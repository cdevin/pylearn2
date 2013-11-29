import theano.tensor as T
from pylearn2.models.mlp import Layer
from pylearn2.utils import serial
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

    def get_output_space(self):
        return self.main_column.get_output_space()

    def fprop(self, state):

        # get the gaters
        z = state
        for i in xrange(len(self.main_column.layers) - 1):
            z = self.main_column.layers[i].fprop(z)

        z = T.switch(T.gt(z,0),1.,0.)

        col_z = []
        for i in xrange(len(self.columns)):
            col_z.append(self.columns[i].fprop(state) * z[:,i].dimshuffle(0, 'x'))

        col_z = T.concatenate(col_z, 1)
        z = T.concatenate([z, col_z], 1)
        print z.tag.test_value.shape


        return z



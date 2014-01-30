from theano import tensor as T
from pylearn2.costs.mlp.dropout import Dropout as DropoutBase
from pylearn2.space import CompositeSpace

class Dropout(DropoutBase):


    def expr(self, model, data, ** kwargs):
        """
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y, cls) = data
        Y_hat = model.dropout_fprop(
            X, cls,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.cost(Y, Y_hat)


    def get_data_specs(self, model):
        """
        .. todo::

            WRITEME
        """
        space = CompositeSpace([model.get_input_space(),
                                model.get_output_space(),
                                model.get_class_space()])
        sources = (model.get_input_source(),
                    model.get_target_source(),
                    model.get_class_source())
        return (space, sources)



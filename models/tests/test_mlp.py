from noisy_encoder.models.mlp_new import DropOutMLP
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from theano import tensor as T

def test_get_weights():
    #Tests that the RBM, when constructed
    #with nvis and nhid arguments, supports the
    #weights interface

    model = DropOutMLP(n_units = [100, 200], input_corruptors = [None, None], hidden_corruptors = [None, None],
            n_outs = 10, act_enc = "sigmoid")
    W = model.get_weights()

def test_get_input_space():
    #Tests that the RBM supports
    #the Space interface

    model = RBM(nvis = 2, nhid = 3)
    space = model.get_input_space()

def test_encode():
    model = DropOutMLP(n_units = [100, 200], input_corruptors = [None, None], hidden_corruptors = [None, None],
            n_outs = 10, act_enc = "sigmoid")
    theano_rng = RandomStreams(17)

    X = T.matrix()

    Y = model(X)



if __name__ == "__main__":
    #test_get_weights()
    test_encode()

from noisy_encoder.models.conv_pylearn import Conv
import theano.sandbox.rng_mrg
RandomStreams = theano.sandbox.rng_mrg.MRG_RandomStreams
from theano import tensor as T

def test_get_weights():
    #Tests that the RBM, when constructed
    #with nvis and nhid arguments, supports the
    #weights interface

    model = RBM(corruptor = None, image_shape = [32, 32],
            nchannels_input = 3, nchannels_outptu = 20, pool_shape = [20, 20], act_enc = "sigmoid")
    W = model.get_weights()

def test_get_input_space():
    #Tests that the RBM supports
    #the Space interface

    model = RBM(nvis = 2, nhid = 3)
    space = model.get_input_space()

def test_gibbs_step_for_v():
    #Just tests that gibbs_step_for_v can be called
    #without crashing (protection against refactoring
    #damage, aren't interpreted languages great?)

    model = RBM(nvis = 2, nhid = 3)

    theano_rng = RandomStreams(17)

    X = T.matrix()

    Y = model.gibbs_step_for_v(X, theano_rng)


if __name__ == "__main__":
    test_get_weights()

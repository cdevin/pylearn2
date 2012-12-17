import theano
from theano import tensor
from pylearn2.utils import serial
from utils.config import get_data_path, get_result_path
import os

def get_score_function(model):

    return  theano.function(inputs = [model.x, model.x_p], outputs = model())




def test(model_path, neutral_image, emot_image):

    model = serial.load(model_path)
    f = get_score_function(model)
    return f(neutral_image, emot_image)



if __name__ == "__main__":

    # load sample images
    data_path = os.path.join(get_data_path(), "faces/TFD/siamese/0/")
    emot = serial.load(data_path + 'test.pkl')
    neutral = serial.load(data_path + 'test_neutral.pkl')

    model_path = os.path.join(get_result_path(), 'naenc/tfd/siamese.pkl')

    print test(model_path, neutral.X[:100], emot.X[:100])

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
    # replace it with what ever test image you have, and you can remove the 4th import line
    data_path = os.path.join(get_data_path(), "faces/TFD/siamese/0/")
    emot = serial.load(data_path + 'test.pkl')
    neutral = serial.load(data_path + 'test_neutral.pkl')

    #model_path = os.path.join(get_result_path(), 'naenc/tfd/siamese.pkl')
    model_path = "/data/lisatmp2/mirzamom/results/naenc/tfd/siamese.pkl"


    # You have to give it 100 images as this kernels are shaped based on batch size.
    # you can just pass 99 garbage + actual image and ignore the the first 99 garbages
    # I will fix this issue later, or set batch size to 1
    print test(model_path, neutral.X[:100], emot.X[:100])

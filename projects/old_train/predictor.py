import argparse
from pylearn2.utils import serial


class Predictor(object):
    def __init__(self, f_single, f_double):
        self.f_single = f_single
        self.f_double = f_double

    def apply(self, input, input2 = None):
        if input2 is None:
            return self.f_single(input)
        else:
            return self.f_double(input, input2)

def make_function(model_path):
    model = serial.load(model_path)
    f_single = model.apply()
    f_double = model.apply(True)

    return f_single, f_double

def save_model(model, save_path):
    serial.save(save_path, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "make a ready independent theano function from model file")
    parser.add_argument('-m', '--model', help = "model file", required = True)
    parser.add_argument('-o', '--output', help = "output file", required = True)
    args = parser.parse_args()

    f_single, f_double = make_function(args.model)
    model = Predictor(f_single, f_double)
    save_model(model, args.output)

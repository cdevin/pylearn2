import numpy
import PIL.Image
from utils import tile_raster_images
from pylearn2.utils import serial



def view_filter(W1,W2):
    filters = []
    W2 = W2.T
    for w2 in W2:
        l2f = numpy.sum(w2*W1,axis=1).reshape((1,-1))
        filters.append(l2f)

    filters = numpy.concatenate(filters)
    return filters



def plot(w):
    w = w.reshape(w.shape[1], 32 * 32, 3)
    x = (w[:,:,0], w[:,:,1], w[:,:,2], None)
#import ipdb
#ipdb.set_trace()


    image = PIL.Image.fromarray(tile_raster_images(X=x,
             img_shape=(32, 32), tile_shape=(40, 40),
             tile_spacing=(1, 1)))


    image.show()
    image.save('test.jpg')



def main(path, layer):

    model = serial.load(path)
    w = model.layers[0].weights.get_value()
    if layer == 2:
        w2 = model.layers[1].weights.get_value()
        w = view_filter(w, w2).T

    print w.shape
    plot(w)

if __name__ == "__main__":

    path = '/RQexec/mirzameh/tmp/naenc/cifar/l1_2_'
    main(path, 2)

import numpy
np = numpy
import scipy.io as io
import Image
import theano
import theano.tensor as T

def gaussian(size, sigma):

    height = float(size)
    width = float(size)
    center_x = width  / 2. + 0.5
    center_y = height / 2. + 0.5

    gauss = np.zeros((height, width))

    for i in xrange(1, int(height) + 1):
        for j in xrange(1, int(height) + 1):
            x_diff_sq = ((float(j) - center_x)/(sigma*width)) ** 2.
            y_diff_sq = ((float(i) - center_y)/(sigma*height)) ** 2.
            gauss[i-1][j-1] = np.exp( - (x_diff_sq + y_diff_sq) / 2.)

    return gauss

def lcn_std_diff(x,size=9):
    p = x.reshape((1,1,48,48))
    #p = (p-T.mean(p))/T.std(p)
    g = gaussian(size,1.591/size)
    g/=g.sum()
    g = np.float32(g.reshape((1,1,size,size)))
    mean = T.nnet.conv.conv2d(p,T.constant(g),
                              (1,1,48,48),
                              (1,1,size,size),
                              'full').reshape((48+size-1,)*2)
    mean = mean[size/2:48+size/2,
                size/2:48+size/2]
    meansq = T.nnet.conv.conv2d(T.sqr(p),T.constant(g),
                                (1,1,48,48),
                                (1,1,size,size),
                                'full').reshape((48+size-1,)*2)
    meansq = meansq[size/2:48+size/2,
                    size/2:48+size/2]
    var = meansq - T.sqr(mean)
    var = T.clip(var, 0, 1e30)
    std = T.sqrt(var)
    std = T.clip(std, T.mean(std), 1e30)
    out = (p - mean) / std
    return out - out.min()

def lcn(x,ishape,size=9):
    """
    expects x to be tensor{3|4}, the first dimension being the number
    of images, and the two last the shape of the image (which should be
    given anyways for optimization purposes
    """
    inshape = (x.shape[0],1,ishape[0],ishape[0])
    p = x.reshape(inshape)
    #p = (p-T.mean(p))/T.std(p)
    g = gaussian(size,1.591/size)
    g/=g.sum()
    g = np.float32(g.reshape((1,1,size,size)))
    mean = T.nnet.conv.conv2d(p,T.constant(g),
                              None,
                              (1,1,size,size),
                              'full').reshape(
                                  (x.shape[0],1)+(ishape[0]+size-1,)*2)
    mean = mean[:,:,
                size/2:ishape[0]+size/2,
                size/2:ishape[1]+size/2]
    v = (p - mean)#.dimshuffle('x','x',0,1)
    var = T.nnet.conv.conv2d(T.sqr(v),T.constant(g),
                             None,
                             (1,1,size,size),
                             'full').reshape(
                                  (x.shape[0],1)+(ishape[0]+size-1,)*2)
    var = var[:,:,
              size/2:ishape[0]+size/2,
              size/2:ishape[1]+size/2]
    std = T.sqrt(var)
    std_mean = T.mean(T.mean(std,axis=3),axis=2).dimshuffle(0,1,'x','x')
    out = v / T.maximum(std,std_mean)
    return (out + 2.5 )/5# - out.min()

if __name__=="__main__":
    data = io.loadmat("/data/lisa/data/faces/TFD/TFD_48x48.mat")
    folds = data['folds']
    images = data['images'][(folds!=0).any(axis=1)]
    labels = data['labs_ex'][(folds!=0).any(axis=1)]

    folds = folds[(folds!=0).any(axis=1)]

    fold =0
    images = images.reshape((images.shape[0],48**2))
    labels = numpy.int8(labels).flatten() - 1
    train, valid, test = [folds[:,fold]==i for i in [1,2,3]]

    train, valid, test =  [[images[train],labels[train]],
                           [images[valid],labels[valid]],
                           [images[test],labels[test]]]

    image = numpy.float32(train[0][0].reshape((48,48)))

    Image.fromarray(image).show(title="pre-lcn")
    import time
    t0 = time.time()
    k = 5
    newimage = numpy.zeros((48,48))

    for i in range(48):
        for j in range(48):
            f = lambda a:min(max(a,0),48)
            sub = image[f(i-k):f(i+k),f(j-k):f(j+k)]
            newimage[i,j]= image[i,j] - sub.mean()#/sub.std()

    print time.time()-t0

    x = T.matrix('x')
    result = lcn_std_diff(x)
    f = theano.function(inputs=[x],outputs=result.reshape((48,48)))
    t0 = time.time()
    z = f(image)
    #z = (z-z.mean())/z.std()
    newimage = z#image-z
    print newimage
    print newimage.mean(),newimage.min(),newimage.max(),newimage.std()
    print time.time()-t0
    Image.fromarray(numpy.uint8((newimage-newimage.min())/(newimage.max()-newimage.min())*255)).show(title="lcn")

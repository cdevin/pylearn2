from scipy.io import loadmat
import numpy
import gc
from preprocessing import lecun_lcn


def lcn_transform(x):
    x =  numpy.transpose(x, axes = [3, 0 , 1, 2])
    for i in xrange(3):
        x[:,:,:,i] = lecun_lcn(x[:,:,:,i], (32,32),7)
    return numpy.transpose(x, axes = [1, 2, 3, 0])


# load difficult train
data = loadmat('/data/lisa/data/SVHN/format2/train_32x32.mat')
valid_index = []
for i in xrange(1, 11):
    index = numpy.nonzero(data['y'] == i)[0]
    index.flags.writeable = 1
    numpy.random.shuffle(index)
    valid_index.append(index[:400])

valid_index = set(numpy.concatenate(valid_index))
train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
valid_index = list(valid_index)
train_index = list(train_index)

train_x = data['X'][:,:,:,train_index]
train_y = data['y'][train_index, :]
valid_x = data['X'][:,:,:,valid_index]
valid_y = data['y'][valid_index, :]


train_size = data['X'].shape[3]
assert train_x.shape[3] == train_size - 4000
assert train_y.shape[0] == train_size - 4000
assert valid_x.shape[3] ==  4000
assert valid_y.shape[0] ==  4000
del data
gc.collect()

# load extra train
data = loadmat('/data/lisa/data/SVHN/format2/extra_32x32.mat')
valid_index = []
for i in xrange(1, 11):
    index = numpy.nonzero(data['y'] == i)[0]
    index.flags.writeable = 1
    numpy.random.shuffle(index)
    valid_index.append(index[:200])

valid_index = set(numpy.concatenate(valid_index))
train_index = set(numpy.arange(data['X'].shape[3])) - valid_index
valid_index = list(valid_index)
train_index = list(train_index)

train_x = numpy.concatenate((train_x, data['X'][:,:,:,train_index]), axis = 3)
train_y = numpy.concatenate((train_y, data['y'][train_index, :]))
valid_x = numpy.concatenate((valid_x, data['X'][:,:,:,valid_index]), axis = 3)
valid_y = numpy.concatenate((valid_y, data['y'][valid_index, :]))

extra_size = data['X'].shape[3]
assert train_x.shape[3] == train_size + extra_size - 6000
assert train_y.shape[0] == train_size + extra_size - 6000
assert valid_x.shape[3] ==  6000
assert valid_y.shape[0] ==  6000

del data
gc.collect()



### lcn
#batch_size = 3000
#for i in xrange(0, new_size, batch_size):
    #stop = -1 if i>= numpy.floor(new_size/ batch_size) * batch_size else i + batch_size
    #print i, stop
    #train_x[:,:,:,i:stop] = lcn_transform(train_x[:,:,:,i:stop])
    #gc.collect()

#for i in xrange(0, 6000, batch_size):
    #stop = -1 if i>= numpy.floor(new_size/ batch_size) * batch_size else i + batch_size
    #print i
    #valid_x[:,:,:,i:stop] = lcn_transform(valid_x[:,:,:,i:stop])


#assert train_x.shape[3] == train_size + extra_size - 6000
#assert train_y.shape[0] == train_size + extra_size - 6000
#assert valid_x.shape[3] ==  6000
#assert valid_y.shape[0] ==  6000


# chunk
new_size = train_x.shape[3] # 598388
path = '/data/lisa/data/SVHN/format2/npy/split/'
count = 0
for i in xrange(0, new_size, 100000):
    stop = -1 if i>= 500000 else i + 100000
    print i, stop
    numpy.save("{}train_32x32_x_{}.npy".format(path, count), train_x[:,:,:,i:stop])
    numpy.save("{}train_32x32_y_{}.npy".format(path, count), train_y[i:stop,:])
    count += 1

numpy.save("{}valid_32x32_x.npy".format(path), valid_x)
numpy.save("{}valid_32x32_y.npy".format(path), valid_y)





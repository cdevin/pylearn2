from pylearn2.utils import serial
data = serial.load('/u/huilgolr/noisylearn/projects/gating2/full/one_billion_train.npy')
old = serial.load('/data/lisa/exp/mirzamom/noisylearn/projects/gating2/one_billion_train.npy')
print len(data)
print len(old)
import numpy as np
print 'in data'
print len(np.where(data==0)[0])
print len(np.where(old==0)[0])
print len(np.where(data==1)[0])
print len(np.where(old==1)[0])

 

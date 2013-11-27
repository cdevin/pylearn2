import numpy as np
import matplotlib.pyplot as plt
import ipdb


conf = np.load('res_out.npy')
labels = np.load('labels.npy')
pred = np.load('pred.npy')


wrong = conf[pred != labels].max(1)
correct = conf[pred == labels].max(1)

ipdb.set_trace()

import numpy as np
import matplotlib.pyplot as plt
import ipdb


conf = np.load('res_out.npy')
labels = np.load('labels.npy')
pred = np.load('pred.npy')
class_conf = np.load('classifier_confidence.npy')


#classifier
class_conf = class_conf.max(1)
wrong = class_conf[pred!=labels]
correct = class_conf[pred==labels]


plt.subplot(2, 1, 1)
plt.hist(correct, range= (0.5,1.))
plt.ylabel('correct')
plt.ylim([0,7000])


plt.subplot(2, 1, 2)
plt.hist(wrong, range= (0.5,1.))
plt.ylabel('wrong')
plt.ylim([0,7000])

plt.savefig('classifier.png')
plt.clf()
#splitter
wrong = conf[pred != labels].max(1)
correct = conf[pred == labels].max(1)

plt.subplot(2, 1, 1)
plt.hist(correct, range= (0.5,1.))
plt.ylabel('correct')
plt.ylim([0,7000])


plt.subplot(2, 1, 2)
plt.hist(wrong, range= (0.5,1.))
plt.ylabel('wrong')
plt.ylim([0,7000])



plt.savefig('splitter.png')
#ipdb.set_trace()
#plt.show()





from pylearn2.datasets.preprocessing import ZCA
from pylearn2.utils import serial

from black_box_dataset import BlackBoxDataset

extra = BlackBoxDataset('extra')

zca = ZCA(filter_bias=.1, n_components = 500)

zca.fit(extra.X)

serial.save('zca.pkl', zca)

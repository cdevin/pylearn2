import argparse
import numpy
from pylearn2.utils import serial
import ipdb

def load(path):

    data = serial.load(path)
    ipdb.set_trace()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'plot monitors')
    parser.add_argument('-f', '--file', help = "monitor file", required = True)

    path =  '/data/lisatmp/mirzamom/jobs/mirzamom_db/conv_google/1/monitor.pkl'
    load(path)



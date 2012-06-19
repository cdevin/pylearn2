import fnmatch
import os
from jobman import api0, sql
from jobman import DD, flatten
import glob
from train import experiment as experiment_train
#from classify import experiment as experiment_class
from svm import experiment as experiment_svm
import numpy
from util.config import DATA_PATH

def explore_train():

    #Database
    TABLE_NAME = 'nac_train_mnist_1'
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)

    #Default values
    state = DD()

    state.dataset = "mnist"
    #state.data_path = DATA_PATH + "TFD/raw/"
    state.data_path = '/RQexec/mirzameh/data/mnist/'
    state.scale = False
    state.nhid = 1024
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.lr_init = 0.09
    state.lr_decay = 2
    state.input_corruption_level = 0.8
    state.hidden_corruption_level = 0.5
    state.l1_l = 0.06
    state.batch_size = 50
    state.n_epochs = 200
    state.norm = False
    state.save_freq = 5
    state.exp_name = 'mnist_l1_1024'


    ind = 0
    for lr in [0.1, 0.01]:
        for in_cr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for hid_cr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                state.lr_init = lr
                state.input_corruption_level = in_cr
                state.hidden_corruption_level = hid_cr
                sql.insert_job(experiment_train, flatten(state), db)
                ind += 1

    db.createView(TABLE_NAME + '_view')
    print ind

#def explore_classify():

    ##Database
    #TABLE_NAME = 'nac_class_2'
    #db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)

    ##Default values
    #state = DD()


    #matches = []
    #for root, dirnames, filenames in os.walk(state.exp_path):
        #for filename in fnmatch.filter(filenames, '*_model.pkl'):
            #matches.append(os.path.join(root, filename))

    #ind = 0
    #for item in matches:
        #state.model_path = item
        #sql.insert_job(experiment_class, flatten(state), db)
        #ind += 1

    #db.createView(TABLE_NAME + '_view')
    #print ind


def explore_svm():

    #Database
    TABLE_NAME = 'nac_svm_mnist_1'
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)

    #Default values
    state = DD()


    state.dataset = "mnist"
    state.model_path = "/RQexec/mirzameh/jobs/mirzamom_db/nac_train_mnist_1/"
    #state.model_path = "/home/mirzameh/projects/noisy_encoder/data/"
    #state.data_path = DATA_PATH + "TFD/raw/"
    state.data_path = '/RQexec/mirzameh/data/mnist/'
    state.scale = False
    state.nhid = 1024
    state.batch_size = 600
    state.c_vals = [1,10, 20]
    state.fold = 0
    state.exp_name = 'mnist_layer1_1024'

    matches = []
    for root, dirnames, filenames in os.walk(state.model_path):
        for filename in fnmatch.filter(filenames, '*_model.pkl'):
            matches.append(os.path.join(root, filename))

    ind = 0
    for item in matches:
        state.model_path = item
        sql.insert_job(experiment_svm, flatten(state), db)
        ind += 1

    db.createView(TABLE_NAME + '_view')
    print ind




if __name__ == "__main__":
    #explore_train()
    explore_svm()

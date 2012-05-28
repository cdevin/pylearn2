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
    TABLE_NAME = 'nac_train_2'
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)

    #Default values
    state = DD()

    state.dataset = "tfd"
    state.data_path = DATA_PATH + "TFD/raw/"
    state.scale = True
    state.nhid = 1024
    state.act_enc = "sigmoid"
    state.act_dec = "sigmoid"
    state.learning_rate = 0.01
    state.input_corruption_level = 0.8
    state.hidden_corruption_level = 0.5
    state.l1_l = 0.06
    state.batch_size = 50
    state.n_epochs = 102
    state.norm = False
    state.save_freq = 2
    state.exp_name = 'tfd_l1'


    ind = 0
    for lr in numpy.logspace(-5, -1, num = 20):
        state.learning_rate = lr
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
    TABLE_NAME = 'nacl2_svm_1'
    db = api0.open_db("postgres://mirzamom:pishy83@gershwin.iro.umontreal.ca/mirzamom_db?table=" + TABLE_NAME)

    #Default values
    state = DD()


    state.dataset = "tfd"
    state.model_path = "/RQexec/mirzameh/jobs/mirzamom_db/nacl2_train_1/"
    state.data_path = DATA_PATH + "TFD/nac_layer1/"
    state.scale = False
    state.nhid = 1024
    state.batch_size = 600
    state.c_vals = [-3,9, 20]
    state.fold = 0
    state.exp_name = 'layer2'

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
    explore_train()
    #explore_svm()

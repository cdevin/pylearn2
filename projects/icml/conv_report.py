import argparse
import pylab
import numpy
from matplotlib import cm
from noisy_encoder.scripts.plot.sql import SQL

def format(results):
    str = "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format("ID", "n_ch_1", "n_ch_2", "n_ch_3", "lr", "decay", "Valid Error", "Test Error")
    for data in results:
        str += "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format(data['id'], data['lr'],  data['hid_corr'],  data['valid error'], data['test error'])

    return str

def reterive_data(experiment, num):
    db = SQL()
    # get classification results

    valid_query = "select {}_view.id, numchannels1, numchannels2, numchannels3, learningrate, decayfactor,\
        {}keyval.fval from {}_view, {}keyval where {}_view.id = dict_id and\
        name = 'valid_score';".format(experiment,
                experiment, experiment, experiment, experiment)

    test_query = "select {}_view.id, {}keyval.fval from {}_view, {}keyval \
            where {}_view.id = dict_id and name = 'test_score';".format(experiment, experiment, experiment, experiment, experiment)
    if num == -1:
        valid_data= db.get_all(valid_query)
        test_data =db.get_all(test_query)
    else:
        valid_data = db.get_many(valid_query, num)
        test_data = db.get_many(test_query, num)

    results = []
    for item in valid_data:
        results.append({'id' : item[0], 'lr' : item[1], 'hid_corr' : item[2], 'valid error' : item[3]})


    for test in test_data:
        for res in results:
            if test[0] == res['id']:
                res['test error'] = test[1]

    results = sorted(results, key = lambda k : k['valid error'])

    return results

def main():
    parser = argparse.ArgumentParser(description = "Pretty print experiment results")
    parser.add_argument('-e', '--experiment', required = True,
            help = "Table name")
    parser.add_argument('-n', '--number', default = 10, type = int,
            help = "max number")
    args = parser.parse_args()


    # report
    results = reterive_data(args.experiment, args.number)
    print format(results)

if __name__ == "__main__":
    main()

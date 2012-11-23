import argparse
import pylab
import numpy
from matplotlib import cm
from noisy_encoder.scripts.plot.sql import SQL

def format(results, which):
    str = "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format("ID", "Learning rate", "Hidden Corruption", "Valid Error", "Test Error")
    for data in results:
        str += "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format(data['id'], data['lr'],  data['hid_corr'],  data['valid error'], data['test error'])

    return str

def reterive_data(experiment, num, fold, which):
    db = SQL()
    # get classification results

    if which == "conv":
        valid_query = "select {}_view.id, lr, mlphiddencorruptionlevels,\
            {}keyval.fval from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score' and fold = {};".format(experiment,
                    experiment, experiment, experiment, experiment, fold)
    else:
        valid_query = "select {}_view.id, lr, hiddencorruptionlevels,\
            {}keyval.fval from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score' and fold = {};".format(experiment,
                    experiment, experiment, experiment, experiment, fold)

    test_query = "select {}_view.id, {}keyval.fval from {}_view, {}keyval \
            where {}_view.id = dict_id and name = 'test_score';".format(experiment, experiment, experiment, experiment, experiment)
    if num == -1:
        valid_data= db.get_all(valid_query)
        test_data =db.get_all(test_query)
    else:
        valid_data = db.get_many(valid_query, num)
        test_data = db.get_many(test_query, num)

    print 'h'
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
    parser.add_argument('-p', '--plot', default = False, action = 'store_true')
    parser.add_argument('-f', '--fold', type=int)
    parser.add_argument('-w', '--which', choices = ['conv', 'siamese'])
    args = parser.parse_args()


    # report
    results = reterive_data(args.experiment, args.number, args.fold, args.which)
    print format(results, args.fold)

if __name__ == "__main__":
    main()

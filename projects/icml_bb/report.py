import argparse
import pylab
import numpy
from matplotlib import cm
from noisy_encoder.scripts.plot.sql import SQL

def format(results):
    str = "{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}\n".format("ID", "units 0", "units 1", "units 2", "pieces 0", "pieces 1", "pieces 2", "lr", "decay", "Valid Error")
    for data in results:
        str += "{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}\n".format(data['id'],
                    data['unit0'],data['unit1'], data['unit2'], data['pieces0'], data['pieces1'],
                    data['pieces2'], data['lr'], data['decay'], data['valid error'])

    return str

def reterive_data(experiment, num):
    db = SQL()
    # get classification results

    valid_query = "select {}_view.id, nunits0, nunits1, nunits2, npieces0, npieces1, npieces2, lrinit, lrdecayfactor,\
            {}keyval.fval from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score';".format(experiment,
                    experiment, experiment, experiment, experiment)

    if num == -1:
        valid_data= db.get_all(valid_query)
    else:
        valid_data = db.get_many(valid_query, num)

    results = []
    for item in valid_data:
        results.append({'id' : item[0], 'unit0' : item[1], 'unit1' : item[2], 'unit2' : item[3], 'pieces0' : item[4],
                        'pieces1' : item[5], 'pieces2': item[6], 'lr': item[7], 'decay':item[8], 'valid error' : item[9]})


    results = sorted(results, key = lambda k : k['valid error'])

    return results

def main():
    parser = argparse.ArgumentParser(description = "Pretty print experiment results")
    parser.add_argument('-e', '--experiment', required = True,
            help = "Table name")
    parser.add_argument('-n', '--number', default = -1, type = int,
            help = "max number")
    args = parser.parse_args()


    # report
    results = reterive_data(args.experiment, args.number)
    print format(results)

if __name__ == "__main__":
    main()

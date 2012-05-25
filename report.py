import psycopg2
import argparse

class SQL():

    def __init__(self):

        self.conn = psycopg2.connect(host = "gershwin.iro.umontreal.ca", database = "mirzamom_db",
                user = "mirzamom", password = "pishy83")
        self.cur = self.conn.cursor()

    def __del__(self):

        self.cur.close()
        self.conn.close()

    def get_many(self, command, num):

        self.cur.execute(command)
        return self.cur.fetchmany(num)

    def get_one(self, command):

        self.cur.execute(command)
        return self.cur.fetchone()


def format(results, params):

    str = "{:^10}|{:^10}|{:^10}|{:^10}\n".format("exp_name", "prob", "Learning rate", "result")
    str+= "---------------------------------------------\n"
    for res, par in zip(results, params):

        str+= "{:<10} |{:^10}|{:^10}|{:^10} \n".format(par[0], par[1], par[2], float(res) * 100)

    return str

def main():

    parser = argparse.ArgumentParser(description = "Pretty print experiment results")
    parser.add_argument('-e', '--experiment', required = True,
            help = "Table name")
    parser.add_argument('-v', '--version', required = True,
            help = "experiment version")
    parser.add_argument('-n', '--number', default = 10, type = int,
            help = "max number")
    args = parser.parse_args()

    db = SQL()
    # get classification results
    table = "{}_svm_{}".format(args.experiment, args.version)
    data = db.get_many("select {}_view.id, modelpath, {}keyval.fval from {}_view, {}keyval \
            where {}_view.id = dict_id and name = 'valid_result' order by \
            fval DESC;".format(table, table, table, table, table), args.number)

    # extract ids and results
    job_ids = [int(item[1].split('/')[-2]) for item in data]
    results = [item[-1] for item in data]

    table = "{}_train_{}".format(args.experiment, args.version)
    commands = ["select expname, prob, learningrate, nhid from {}_view \
            where id = {}".format(table, id)  for id in job_ids]
    params = [db.get_one(item) for item in commands]


    print format(results, params)






if __name__ == "__main__":
    main()

import psycopg2
import argparse
import pylab
import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

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

    def get_all(self, command):

        self.cur.execute(command)
        return self.cur.fetchall()



def format(results, params, iter_num, job_ids):

    str = "{:^15}|{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^20}\n".format("exp_name", "job_id", "iter num", "input corr", "hidden corr", "learning rate", "result")
    str+= "-------------------------------------------------------------------------------------------------------\n"

    for res, par, iter, id in zip(results, params, iter_num, job_ids):

        str+= "{:<14} |{:^10}|{:^10}|{:^10}|{:^10}|{:^10}|{:^20} \n".format(par[0], id,  iter,  par[1], par[2], par[3], float(res) * 100)

    return str


def plot_lr(performance, learning_rate, name):


    pylab.scatter(learning_rate, performance)
    #pylab.xscale('log')
    pylab.show()
    pylab.savefig(name)


def plot_scatter(performance, input_corr, hidd_corr, name):

    # scatter
    fig = pylab.figure()

    x_l = numpy.linspace(0,0.9, 10)
    y_l = numpy.linspace(0,0.9, 10)
    z_l = numpy.zeros((10,10))

    for x_ind, x in enumerate(x_l):
        for y_ind, y in enumerate(y_l):
            per = [0.]
            for in_c, hid_c, perf in zip(input_corr, hidd_corr, performance):
                if numpy.allclose(in_c, x) and numpy.allclose(hid_c, y):
                    per.append(perf)
            z_l[x_ind, y_ind] = numpy.max(per)


    ax = fig.add_subplot(2,1,1)
    ax.plot(x_l, z_l.max(1))
    ax.set_xlim((0, 0.9))
    #ax.set_ylim(0.9814, .984)
    ax.set_xlabel('Input Corruption')
    ax = fig.add_subplot(2,1,2)
    ax.plot(y_l, z_l.max(0))
    ax.set_xlim((0, 0.9))
    ax.set_xlabel('Hidden Corruption')


    #x_l, y_l = numpy.meshgrid(x_l, y_l)
    #x_l = x_l.ravel()
    #y_l = y_l.ravel()
    #z_l = z_l.ravel()

    #ax = fig.add_subplot(3,2, 5)
    #ax.hexbin(x_l, y_l, C = z_l, cmap=cm.jet, bins=None)
    #ax.set_ylim((0, 1))
    #ax.set_ylabel('Input Corruption')
    #ax.set_xlim((0, 1))
    #ax.set_xlabel('Hidden Corruption')

    pylab.show()
    pylab.savefig(name)


#def plot_scatter(performance, input_corr, hidd_corr, name):

    ## scatter
    #fig = pylab.figure()
    #ax = fig.add_subplot(3,2,1)
    #ax.scatter(input_corr, performance)
    #ax.set_xlabel('Input Corruption')
    #ax.set_xlim((0, 1))
    #ax = fig.add_subplot(3,2,3)
    #ax.scatter(hidd_corr, performance)
    #ax.set_xlabel('Hidden Corruption')
    #ax.set_xlim((0, 1))

    #x_l = numpy.linspace(0,0.9, 10)
    #y_l = numpy.linspace(0,0.9, 10)
    #z_l = numpy.zeros((10,10))

    #for x_ind, x in enumerate(x_l):
        #for y_ind, y in enumerate(y_l):
            #per = [0.]
            #for in_c, hid_c, perf in zip(input_corr, hidd_corr, performance):
                #if numpy.allclose(in_c, x) and numpy.allclose(hid_c, y):
                    #per.append(perf)
            #z_l[x_ind, y_ind] = numpy.max(per)


    #ax = fig.add_subplot(3,2,2)
    #ax.plot(x_l, z_l.max(1))
    #ax.set_xlim((0, 0.9))
    #ax.set_xlabel('Input Corruption')
    #ax = fig.add_subplot(3,2,4)
    #ax.plot(y_l, z_l.max(0))
    #ax.set_xlim((0, 0.9))
    #ax.set_xlabel('Hidden Corruption')


    #x_l, y_l = numpy.meshgrid(x_l, y_l)
    #x_l = x_l.ravel()
    #y_l = y_l.ravel()
    #z_l = z_l.ravel()

    #ax = fig.add_subplot(3,2, 5)
    #ax.hexbin(x_l, y_l, C = z_l, cmap=cm.jet, bins=None)
    #ax.set_ylim((0, 1))
    #ax.set_ylabel('Input Corruption')
    #ax.set_xlim((0, 1))
    #ax.set_xlabel('Hidden Corruption')

    #pylab.show()
    #pylab.savefig(name)

def reterive_data(experiment, version, num):

    db = SQL()
    # get classification results
    table = "{}_svm_{}".format(experiment, version)

    if num == -1:
        data = db.get_all("select {}_view.id, modelpath, {}keyval.fval from {}_view, {}keyval \
            where {}_view.id = dict_id and name = 'valid_result' order by \
            fval DESC;".format(table, table, table, table, table))
    else:
        data = db.get_many("select {}_view.id, modelpath, {}keyval.fval from {}_view, {}keyval \
            where {}_view.id = dict_id and name = 'valid_result' order by \
            fval DESC;".format(table, table, table, table, table), num)


    # extract ids and results
    job_ids = [int(item[1].split('/')[-2]) for item in data]
    results = [item[-1] for item in data]
    iter_num = [item[1].split('/')[-1].split('_')[-2] for item in data]

    table = "{}_train_{}".format(experiment, version)
    commands = ["select expname, inputcorruptionlevel, hiddencorruptionlevel, learningrate, nhid from {}_view \
            where id = {}".format(table, id)  for id in job_ids]
    params = [db.get_one(item) for item in commands]

    return params, results, job_ids, iter_num


def main():

    parser = argparse.ArgumentParser(description = "Pretty print experiment results")
    parser.add_argument('-e', '--experiment', required = True,
            help = "Table name")
    parser.add_argument('-v', '--version', required = True,
            help = "experiment version")
    parser.add_argument('-n', '--number', default = 10, type = int,
            help = "max number")
    parser.add_argument('-p', '--plot', default = False, action = 'store_true')
    args = parser.parse_args()


    # report
    params, results, job_ids, iter_num = reterive_data(args.experiment, args.version, args.number)
    print format(results, params, iter_num, job_ids)

    # plot
    if args.plot == True:
        params, results, job_ids, iter_num = reterive_data(args.experiment, args.version, -1)
        name = "{}_{}_corr.png".format(args.experiment, args.version)
        plot_scatter(results, [item[1] for item in params], [item[2] for item in params], name)
        name = "{}_{}_lr.png".format(args.experiment, args.version)
        plot_lr(results, [item[-2] for item in params], name)



if __name__ == "__main__":
    main()

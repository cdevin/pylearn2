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


def format(results, smooth):
    if smooth:
        str = "{:^10}|{:^15}|{:^30}|{:^30}|{:^20}|{:^20}\n".format("ID", "Learning rate", "Activation", "Gaussian Corruption", "Binomial Corruption", "Test Error")
        for data in results:
            str += "{:^10}|{:^15}|{:^20}|{:^30}|{:^30}|{:^20}\n".format(data['id'], data['lr'], data['act_enc'], data['gauss_corr'], data['bi_corr'], data['test error'])
    else:
        str = "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format("ID", "Learning rate", "Activation", "Corruption Levels", "Test Error")
        for data in results:
            str += "{:^10}|{:^15}|{:^20}|{:^30}|{:^20}\n".format(data['id'], data['lr'], data['act_enc'], data['corruptions'], data['test error'])

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


def plot_group(performance, input_corr, hidd_corr, group_nums, name):

    uni_gr = numpy.unique(group_nums)

    x_l = numpy.linspace(0,0.9, 10)
    y_l = numpy.linspace(0,0.9, 10)
    z_l = numpy.zeros((len(uni_gr), 10,10))

    fig = pylab.figure()
    ax_in = fig.add_subplot(2,1,1)
    ax_hid = fig.add_subplot(2,1,2)

    for i, gr_num in enumerate(uni_gr):
        new_in_corr = []
        new_hid_corr = []
        new_perf = []

        for in_cr, hid_cr, gr, per in zip(input_corr, hidd_corr, group_nums, performance):
            if gr == gr_num:
                new_in_corr.append(in_cr)
                new_hid_corr.append(hid_cr)
                new_perf.append(per)

        for x_ind, x in enumerate(x_l):
            for y_ind, y in enumerate(y_l):
                per = [0.]
                for in_c, hid_c, perf in zip(new_in_corr, new_hid_corr, new_perf):
                    if numpy.allclose(in_c, x) and numpy.allclose(hid_c, y):
                        per.append(perf)
                z_l[i, x_ind, y_ind] = numpy.max(per)



        ax_in.plot(x_l, z_l[i].max(1), color = cm.Accent(i/float(len(uni_gr)),1), label = str(gr_num))
        ax_hid.plot(x_l, z_l[i].max(0), color = cm.Accent(i/float(len(uni_gr)),1), label = str(gr_num))

    ax_in.set_xlim((0, 0.9))
    ax_in.set_xlabel('Input Corruption')
    handles, labels = ax_in.get_legend_handles_labels()
    ax_in.legend(handles[::-1], labels[::-1], loc = 3)

    ax_hid.set_xlim((0, 0.9))
    ax_hid.set_xlabel('Hidden Corruption')
    handles, labels = ax_hid.get_legend_handles_labels()
    ax_hid.legend(handles[::-1], labels[::-1], loc = 3)

    pylab.show()
    pylab.savefig(name)

def reterive_data(experiment, num, smooth = False):

    db = SQL()
    # get classification results

    if smooth:
        valid_query = "select {}_view.id, lr, actenc,\
            gaussiancorruptionlevels, binomialcorruptionlevels, {}keyval.fval\
            from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score';".format(experiment, experiment, experiment, experiment, experiment)
    else:
        valid_query = "select {}_view.id, lr, actenc,\
            corruptionlevels,  {}keyval.fval\
            from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score';".format(experiment, experiment, experiment, experiment, experiment)
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
        results.append({'id' : item[0], 'lr' : item[1], 'act_enc' : item[2], 'gauss_corr' : item[3], 'bi_corr': item[4], 'valid error' : item[5]})

    for test in test_data:
        for res in results:
            if test[0] == res['id']:
                res['test error'] = test[1]

    results = sorted(results, key = lambda k : k['test error'])

    return results


def main():

    parser = argparse.ArgumentParser(description = "Pretty print experiment results")
    parser.add_argument('-e', '--experiment', required = True,
            help = "Table name")
    parser.add_argument('-n', '--number', default = 10, type = int,
            help = "max number")
    parser.add_argument('-p', '--plot', default = False, action = 'store_true')
    parser.add_argument('-g', '--group', default = False, action = 'store_true')
    parser.add_argument('-s', '--smooth', default = False, action = 'store_true')
    args = parser.parse_args()


    # report
    results = reterive_data(args.experiment, args.number, args.smooth)
    print format(results, args.smooth)

    # plot
    if args.plot == True:
        if args.group == True:
            params, results, job_ids, iter_num = reterive_data(args.experiment, args.version, -1, args.group)
            name = "{}_{}_corr.png".format(args.experiment, args.version)
            plot_group(results, [item[1] for item in params], [item[2] for item in params], [item[3] for item in params], name)
        else:
            params, results, job_ids, iter_num = reterive_data(args.experiment, args.version, -1, False)
            name = "{}_{}_corr.png".format(args.experiment, args.version)
            plot_scatter(results, [item[1] for item in params], [item[2] for item in params], name)
            name = "{}_{}_lr.png".format(args.experiment, args.version)
            plot_lr(results, [item[-2] for item in params], name)



if __name__ == "__main__":
    main()

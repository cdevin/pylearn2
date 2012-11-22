import argparse
import pylab
import numpy
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from noisy_encoder.scripts.plot.sql import SQL

def format(results, which):
    if which == 'smooth':
        str = "{:^10}|{:^15}|{:^30}|{:^30}|{:^20}|{:^20}\n".format("ID", "Learning rate", "Activation", "Gaussian Corruption", "Binomial Corruption", "Test Error")
        for data in results:
            str += "{:^10}|{:^15}|{:^20}|{:^30}|{:^30}|{:^20}\n".format(data['id'], data['lr'], data['act_enc'], data['gauss_corr'], data['bi_corr'], data['test error'])
    elif which == 'group':
        str = "{:^10}|{:^15}|{:^30}|{:^30}|{:^30}|{:^30}|{:^30}|{:^20}|{:^20}\n".format("ID",
                "Learning rate", "Activation", "Activation L1", "Gaussian Corruption",
                "Binomial Corruption", "Group Corruption", "Group Size", "Test Error")
        for data in results:
            str += "{:^10}|{:^15}|{:^30}|{:^30}|{:^30}|{:^30}|{:^30}|{:^20}|{:^20}\n".format(data['id'],
                    data['lr'], data['act_enc'], data['act_l1'], data['gauss_corr'], data['bi_corr'],
                    data['group_corr'], data['group_size'], data['test error'])
    elif which == 'normal':
        str = "{:^10}|{:^15}|{:^30}|{:^20}|{:^20}\n".format("ID", "Learning rate", "Activation", "Corruption Levels", "Test Error")
        for data in results:
            str += "{:^10}|{:^15}|{:^20}|{:^30}|{:^20}\n".format(data['id'], data['lr'], data['act_enc'], data['corruptions'], data['test error'])
    else:
        raise NameError('Unknown experiment type: {}'.format(which))

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

def reterive_data(experiment, num, which):

    db = SQL()
    # get classification results

    if which == 'smooth':
        valid_query = "select {}_view.id, lr, actenc,\
            gaussiancorruptionlevels, binomialcorruptionlevels, {}keyval.fval\
            from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score';".format(experiment, experiment, experiment, experiment, experiment)
    elif which == 'group':
        valid_query = "select {}_view.id, lr, actenc, actl1ratio,\
            gaussiancorruptionlevels, binomialcorruptionlevels, groupcorruptionlevels,\
            groupsizes, {}keyval.fval from {}_view, {}keyval where {}_view.id = dict_id and\
            name = 'valid_score';".format(experiment, experiment, experiment, experiment, experiment)
    elif which == 'normal':
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
        if which == 'smooth':
            results.append({'id' : item[0], 'lr' : item[1], 'act_enc' : item[2], 'gauss_corr' : item[3], 'bi_corr': item[4], 'valid error' : item[5]})
        elif which == 'group':
            results.append({'id' : item[0], 'lr' : item[1], 'act_enc' : item[2], 'act_l1' : item[3],
                'gauss_corr' : item[4], 'bi_corr': item[5], 'group_corr': item[6], 'group_size': item[7], 'valid error' : item[8]})


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
    parser.add_argument('-w', '--which', help = "Which experimnet type", choices = ['normal', 'smooth', 'group'])
    args = parser.parse_args()


    # report
    results = reterive_data(args.experiment, args.number, args.which)
    print format(results, args.which)


if __name__ == "__main__":
    main()

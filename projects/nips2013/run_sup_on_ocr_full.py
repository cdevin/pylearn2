import sys
import os
decapitate = "0"
batch_size = "100"
try:
    _, path = sys.argv
except:
    try:
        _, path, decapitate = sys.argv
    except:
        _, path, decapitate, batch_size = sys.argv

assert ('best' in path) or ('cont' in path) or ('retrain' in path)

parent = path.split('/')[:-1]
parent = '/'.join(parent)

if parent != '':
    parent = parent + '/'

outpath = parent + 'sup_on_full_'+path.split('/')[-1]
outpath = outpath.replace('.pkl','_' + decapitate + '_' + batch_size + '.yaml')
print "THEANO_FLAGS='device=gpu' train.py",outpath
if os.path.exists(outpath.replace('.yaml','.pkl')):
    print outpath.replace('.yaml', '.pkl')
    assert False

f = open(outpath, 'w')

f.write(
"""
!obj:pylearn2.train.Train {
    dataset:  &train !obj:pylearn2.datasets.ocr.OCR {
        which_set: "train_all",
        one_hot: 1,
    },
        model: !obj:galatea.dbm.inpaint.super_dbm.MLP_Wrapper {
                        decapitate: %(decapitate)s,
                        super_dbm: !obj:galatea.dbm.inpaint.super_dbm.set_niter {
                                super_dbm: !pkl: "%(path)s",
                                niter: 5
                        },
    },
    algorithm: !obj:pylearn2.training_algorithms.bgd.BGD {
               line_search_mode: 'exhaustive',
               batch_size: %(batch_size)s,
               set_batch_size: 1,
               updates_per_batch: 3,
               reset_alpha: 0,
               conjugate: 1,
               reset_conjugate: 0,
               monitoring_dataset: {
                                'train' : *train,
                                'valid' : !obj:pylearn2.datasets.ocr.OCR {
                                        which_set: "valid",
                                        one_hot: 1,
                                        },
                                'test' : !obj:pylearn2.datasets.ocr.OCR {
                                        which_set: "test",
                                        one_hot: 1,
                                        }
               },
               cost : !obj:galatea.dbm.inpaint.super_dbm.SuperDBM_ConditionalNLL {
               },
               termination_criterion: !obj:pylearn2.termination_criteria.MonitorBased
               {
                        channel_name: "valid_err",
                        prop_decrease: .000,
                        N : 50
               }
        },
    extensions: [
                !obj:pylearn2.train_extensions.best_params.MonitorBasedSaveBest {
                        channel_name: "valid_err",
                        save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}_best.pkl"
                }
        ],
    save_path: "${PYLEARN2_TRAIN_FILE_FULL_STEM}.pkl",
    save_freq : 1
}
""" % locals())

f.close()
print 'run train yourself, for some reason the subprocess was claiming a second gpu'
quit()
from pylearn2.utils.shell import run_shell_command
print 'shell will hide output until done, yay python'
out, rc = run_shell_command('train.py '+outpath)
print 'return code ',rc
print out

import get_dataset
import ipdb as pdb
import numpy as np
from textmenu import textmenu
import matplotlib.pyplot as plt
import matplotlib as mplt

mplt.rc('xtick', labelsize=22)
mplt.rc('ytick', labelsize=22)

def batchboost_n_samples_to_n_preds(K, v_n_samples):
    l = len(v_n_samples)
    delt_v_nsamples = v_n_samples[1:] - v_n_samples[0:-1]
    delt_v_nsamples = np.hstack([v_n_samples[0], delt_v_nsamples])
    return (np.cumsum((np.arange(l) // K + 1) * delt_v_nsamples) + v_n_samples*2)

def deepboost_n_samples_to_n_preds(N, v_n_samples):
    return v_n_samples * N *3

def average_end_at(arr, indices, l=5):
    return np.array([ arr[ind-l:ind].mean() for ind in indices ])

def dbfname_to_plot_points(fname, N, traval, col):
    d = np.load(fname)
    d_n_preds = deepboost_n_samples_to_n_preds(N, d[traval][:, 0])
    Kd = d[traval].shape[0] // 50
    d_select_indices = np.arange(0, d_n_preds.shape[0], Kd) + Kd-1
    if d_select_indices[-1] > d_n_preds.shape[0]-1:
        d_select_indices[-1] = d_n_preds.shape[0] -1
    return d_n_preds[d_select_indices], average_end_at(d[traval][:,col], d_select_indices, 5)


datasets = get_dataset.all_names()
indx = textmenu(datasets)
if indx is None:
    exit(0)
dataset = datasets[indx]
x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
model_name_suffix = dataset

Kdb = None
if dataset == 'a9a':
    n_nodes = 8
    col = 1
elif dataset == 'mnist':
    n_nodes = 5
    cost_multiplier = 1
    col = 1
elif dataset == 'slice':
    n_nodes = 7
    col = 2
elif dataset == 'year':
    n_nodes = 10
    col = 2
    Kdb = 1000
    max_db_select = n_nodes
elif dataset == 'abalone':
    n_nodes = 8
    col = 2

db_log = '../log/err_vs_gstep_{:s}.npz'.format(model_name_suffix)
bb_log = '../log/batch_err_vs_gstep_{:s}.npz'.format(model_name_suffix)

db = np.load(db_log)
bb = np.load(bb_log)

traval = 'val_err'    #train_err or val_err
plot_all_pts = False  # print convergence of each weak learner in batch setting 
plot_legend = True    # plot legend?  only abalone prints legend
ABALONE_VARY_N = False # plot vary N=? on abalone set. 
results_only = False  # print no plots, and only print final result from logs. 

K = bb[traval].shape[0] // n_nodes
if dataset=='mnist':
    K=2039
bb_n_preds = batchboost_n_samples_to_n_preds(K, bb[traval][:, 0])

print 'Results: Base {:f}, GB {:f}, SGB {:f}'.format(bb[traval][K-5:K,col].mean(), bb[traval][-5:, col].mean(), db[traval][-5:,col].mean()) 
if results_only:
    import sys
    sys.exit(0)


print K
bb_select_indices = np.arange(0, bb_n_preds.shape[0], K) + K-1
if bb_select_indices[-1] >= bb_n_preds.shape[0]:
    bb_select_indices[-1] = bb_n_preds.shape[0] - 1
print bb_select_indices
bb_select_indices = bb_select_indices[:-1]

if plot_all_pts:
    plt.loglog(bb_n_preds, bb[traval][:,col], linewidth=7, label='Batch_all')
#plt.loglog(db_n_preds, db[traval][:,3-col], label = 'streaming_all_other_col')
plt.loglog(bb_n_preds[bb_select_indices], average_end_at(bb[traval][:,col], bb_select_indices, 5), marker='o', markersize=12, linewidth=7,label='Batch')

if not ABALONE_VARY_N:
    x,y = dbfname_to_plot_points(db_log, n_nodes, traval, col)
    plt.loglog(x, y, marker='', markersize=12, linewidth=3, label='Streaming')

else: 

    N=8
    x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_8.npz', N, traval, col)
    plt.loglog(x,y,linewidth=3, label='N=8')

    N=12
    x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_12.npz', N, traval, col)
    plt.loglog(x,y,linewidth=3, label='N=12')

    N=16
    x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_16.npz', N, traval, col)
    plt.loglog(x,y,linewidth=3, label='N=16')

    N=20
    x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_20.npz', N, traval, col)
    plt.loglog(x,y,linewidth=3, label='N=20')
    
    N=3
    x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_3.npz', N, traval, col)
    plt.loglog(x[3:],y[3:],linewidth=3, label='N=3')

    #N=24
    #x, y = dbfname_to_plot_points('../log/err_vs_gstep_abalone_24.npz', N, traval, col)
    #plt.loglog(x,y,linewidth=7, label='N=24')


if plot_legend and dataset == 'abalone':
    plt.legend(loc='best', fontsize=22)

plt.show(block=False)
if ABALONE_VARY_N:
    plt.savefig('../plot/aistats2017/abalone_vary_n.pdf')
else:
    plt.savefig('../plot/aistats2017/{:s}.pdf'.format(model_name_suffix))

pdb.set_trace()

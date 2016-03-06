#!/usr/bin/env python

from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import numpy as np
import numdifftools as nd

from timer import Timer


from collections import namedtuple
DataPt = namedtuple('DataPt', ['x','y'])

from IPython import embed

# from sgd_fast sklearn source
class SquaredLoss(object):
    """Squared loss traditional used in linear regression."""
    def loss(self, p, y):
        return 0.5 * (p - y) * (p - y)
    def dloss(self, p, y):
        """ p is the prediction, y is the true value"""
        return p - y

class EpsilonInsensitive(object):
    """Epsilon-Insensitive loss (used by SVR).  loss = max(0, |y - p| - epsilon) """
    def __init__(self, epsilon):
        self.epsilon = epsilon
    def loss(self, p, y):
        ret = abs(y - p) - self.epsilon
        return ret if ret > 0 else 0
    def dloss(self, p, y):
        if y - p > self.epsilon:
            return -1.
        elif p - y > self.epsilon:
            return 1.
        else:
            return 0.

SqLoss = SquaredLoss()
EpsInsensitive = EpsilonInsensitive(1e-2)

def linear_features(w, x):
    return w.dot(x)
def square_features(w, x):
    return w.dot(x*x)

class Node(object):
    def __init__(self, loss_obj, parent=None, name="NoName", input_dim = 0,
            predict_func=linear_features):
        self.parent = parent
        self.name = name
        self.loss_obj = loss_obj
        self.w = np.zeros(input_dim)
        self.predict_func = predict_func
    def predict(self, x):
        return self.predict_func(self.w, x)
    def dloss(self, val, true_val):
        # need to compute the gradient wrt parent also
        if self.parent is None:
            return self.loss_obj.dloss(val, true_val)
        else:
            # children nodes want to predict the -gradient of the parent
            #return self.loss_obj.dloss(val, -self.parent.dloss(val, true_val))
            return self.parent.dloss(val, true_val)
    def loss(self, val, true_val):
        return self.loss_obj.loss(val, true_val)
    def grad_step(self, x, loss, step_size):
        pred = partial(self.predict_func, x=x)
        param_grad_func = nd.Derivative(pred)
        grad = param_grad_func(self.w)
        self.w = self.w - step_size*grad*loss
        return self.w


feature_dim = 1
def f(x):
    #return 0.25*x
    #return 1e-2*x*x  + 5.00*x
    #return 1e0*x*x  + 5.00*x
    return 8.*cos(x) + 2.5*x*sin(x) + 2.8*x

# dataset generator function
def dataset(num_pts, seed=0):
    max_x = 5
    np.random.seed(seed)
    for _ in xrange(num_pts):
        x = -max_x+2.*max_x*np.random.rand(1)[0]
        y = f(x)
        yield DataPt(x,y)

def compute_running_average(nodes, predictions):
    num_nodes = len(nodes)
    weights = [2./(i+1.) for i in xrange(1, num_nodes+1)]
    partial_sums = np.zeros(num_nodes) 
    partial_sums[0] = predictions[0]
    for i in xrange(1,num_nodes):
        partial_sums[i] = (1.-weights[i]) * partial_sums[i-1] + weights[i]*predictions[i]
    return partial_sums

def predict_layer(pt, child_nodes, top_node):
    """
    :param pt - DataPt object that has .x and .y
    """
    # predict up
    predictions = np.array([node.predict(pt.x) for node in child_nodes])
    #helper = lambda node : node.predict(pt.x)
    #predictions = np.array(p.map(helper, child_nodes))
    # compute running average
    partial_sums = compute_running_average(child_nodes, predictions)
    top_loss = top_node.loss(partial_sums[-1], pt.y)
    return partial_sums, top_loss

def main(num_pts, num_children, learning_rate=1.5, learning_scale = 0.8, rand_seed=0):
    top_node= Node(SqLoss, parent=None, name="root", input_dim = 0)
    child_nodes= [Node(SqLoss, parent=top_node, input_dim = feature_dim, name='Child {:d}'.format(i))\
            for i in xrange(num_children)]
    #child_nodes = []
    #for i in xrange(num_children):
    #    func = linear_features
    #    if i % 2 == 0:
    #        func = square_features 
    #    child_nodes.append(Node(None, parent=top_node, input_dim=feature_dim, predict_func=func,
    #        name='Child {:d}'.format(i))) 
        
    validation_set = [pt for pt in dataset(500, seed=rand_seed+1)]

    batch_set = [pt for pt in dataset(num_pts, seed=rand_seed)]
    from sklearn.linear_model.ridge import Ridge
    batch_learner = Ridge(alpha=1e-15, fit_intercept=False) 
    batch_learner.fit(np.vstack([pt.x for pt in batch_set]), np.array([pt.y for pt in batch_set]))
    batch_pred = batch_learner.predict(np.vstack([pt.x for pt in validation_set]))
    Yval = np.array([pt.y for pt in validation_set])
    err = batch_pred - Yval; mean_batch_err = np.mean(np.sqrt(err*err))
    print('Batch err: {:.4g}'.format(mean_batch_err))


    npprint = partial(np.array_str, precision = 3)

    multiprocess = num_children >= 75
    if multiprocess:
        from pathos.multiprocessing import ProcessingPool as Pool
        from pathos.multiprocessing import cpu_count
        p = Pool(int(ceil(0.75*cpu_count())))
        val_helper = partial(predict_layer, child_nodes=child_nodes, top_node=top_node)

    learner_weights = np.array([node.w for node in child_nodes])
    disp_num_child = 15
    if num_children < disp_num_child:
        print('Child learner weights: {}'.format(npprint(learner_weights.ravel())))

    validation_preds = []
    per_iter_learner_weights =  []
    print 'Starting Online Boosting...'
    for i,pt in enumerate(dataset(num_pts, seed=rand_seed)):
        per_iter_learner_weights.append(learner_weights)
        # Compute loss on Validation set
        if multiprocess:
            val_results = p.map(val_helper, validation_set); 
        else:
            val_results = [predict_layer(val_pt, child_nodes, top_node) for val_pt in validation_set]
        val_psums, val_losses = zip(*val_results); 
        val_preds = [psum[-1] for psum in val_psums]; validation_preds.append(val_preds)
        avg_val_loss = np.mean(val_losses)
        # Compute the partial sums, loss on current data point 
        partial_sums, top_loss = predict_layer(pt, child_nodes, top_node)

        # get the gradient of the top loss at each partial sum
        true_val = pt.y 
        offset_partials = partial_sums.copy()
        offset_partials[1:] = partial_sums[:-1]
        offset_partials[0] = 0
        dlosses = [node.dloss(pred_val, true_val) for pred_val,node in zip(offset_partials, child_nodes)]
        step_size = learning_scale * 1./np.power((i+1), learning_rate)
        learner_weights = np.array([node.grad_step(pt.x, loss, step_size)\
                for (node, loss) in zip(child_nodes, dlosses)])
        if  i < 1 or i == num_pts-1 or (i < num_children and num_children < disp_num_child)\
                or i % min(int(ceil(num_pts*0.05)),25) == 0:
            print('Iteration {:d}/{:d}: (x={:.2g},y={:.2g})'.format(i+1, num_pts, pt.x, pt.y))
            print(' Avg validation loss on pt: {:.4g} vs Batch: {:.4g}'.format(avg_val_loss,
                mean_batch_err))
            print('  Top layer loss on pt: {:.4g}'.format(top_loss))
            if num_children < disp_num_child:
                print('  Child learner weights: {}'.format(npprint(learner_weights.ravel())))
                print('  Partial sums: {}'.format(npprint(partial_sums)))
            print('  Took descent step of step size {:.4g}...'.format(step_size))
    #endfor

    return validation_set, validation_preds, batch_pred, batch_set, per_iter_learner_weights


def plot_per_iter_results(validation_pts, validation_preds, batch_set, batch_pred, learner_weights ):
    import time,os
    num_iters = len(validation_preds)

    val_x = np.array([pt.x for pt in validation_pts])
    val_y = np.array([pt.y for pt in validation_pts])

    pred_base_outdir = os.path.join(os.path.split(__file__)[0], '../predict_func') 
    if not os.path.isdir(pred_base_outdir):
        os.mkdir(pred_base_outdir)

    y_lims = [min(np.min(val_y), np.min(validation_preds)), max(np.max(val_y), np.max(validation_preds))]
    x_lims = [np.min(val_x), np.max(val_x)]
    axis_lims = list(x_lims); axis_lims.extend(y_lims)   
    axis_lims[0] = axis_lims[0]-0.05*abs(axis_lims[0]); axis_lims[2] = axis_lims[2]-0.05*abs(axis_lims[2]);
    axis_lims[1] = axis_lims[1]+0.05*abs(axis_lims[1]); axis_lims[3] = axis_lims[3]+0.05*abs(axis_lims[3]);

    val_sort = np.argsort(val_x)
    for i in xrange(num_iters):
        print('Drawing frame {}/{}'.format(i+1, num_iters))
        plt.figure(1)
        plt.clf()
        plt.plot(val_x[val_sort], val_y[val_sort], label='Ground Truth', linewidth=3)
        plt.plot(val_x[val_sort], batch_pred[val_sort], '--', linewidth=2, label='Batch Learner')
        plt.plot(val_x[val_sort], np.array(validation_preds[i])[val_sort], linewidth=1.5, label='Online Boosting')
        if i > 0:
            plt.plot(batch_set[i].x, batch_set[i].y, 's', markersize=8, linewidth=1.5, label='Point Recieved')
        plt.xlabel('X', fontsize=12); plt.ylabel('Y', fontsize=12)
        plt.title('Data Pt. {}/{}'.format(i, num_iters), fontsize=12)
        plt.legend(loc=2,fontsize=12)
        plt.axis(axis_lims)
        out_fname = os.path.join(pred_base_outdir, 'frame{:03d}.png'.format(i))
        plt.savefig(out_fname, format='png')
    #endfor
    print('Saved out: {}'.format(out_fname))

    learner_out_dir = os.path.join(os.path.split(__file__)[0], '../') 
    num_learners = len(learner_weights[0])
    plt.figure(2)
    plt.clf()
    weights = np.vstack([w.ravel() for w in learner_weights])
    plt.plot(np.arange(num_iters), weights)
    plt.xlabel('Num. Data Points', fontsize=12)
    plt.ylabel('Linear Coefficient', fontsize=12)
    plt.title('Learner Weight vs Num Data Pt.', fontsize=12)
    labels = ['Learner {:d}'.format(n) for n in xrange(num_learners)]
    plt.legend(labels, fontsize=8)
    out_fname = os.path.join(learner_out_dir, 'LearnerWeightVsTime.pdf')
    plt.savefig(out_fname, format='pdf')
    print('Saved out: {}'.format(out_fname))
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("1d Online Gradient Boosting")
    parser.add_argument('-n', '--num_pts', default=4, help='number of data points to run')
    parser.add_argument('-c', '--num_children', default=3, help='number of children learners')
    parser.add_argument('-l', '--learning_rate', default=3.0, help='eta in step_size = s * 1/(i+1)^eta')
    parser.add_argument('-ls', '--learning_scale', default=0.8, help='s in step_size = s * 1/(i+1)^eta')
    parser.add_argument('-r', '--rand_seed', default=0, help='random seed for data gen')
    parser.add_argument('-s', '--save', action='store_true', help='save frames of predictions versus time, learner weights vs time')
    args = parser.parse_args()
    num_pts     = int(args.num_pts)
    num_child   = int(args.num_children)
    rand_seed   = int(args.rand_seed)
    learning_rate = float(args.learning_rate)
    learning_scale = float(args.learning_scale)

    validation_pts, validation_preds, batch_pred, batch_set, learner_weights = main(num_pts, num_child, 
            learning_rate, learning_scale, rand_seed)
    if args.save:
        plot_per_iter_results(validation_pts, validation_preds, batch_set, batch_pred, learner_weights)







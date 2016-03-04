#!/usr/bin/env python

from collections import namedtuple
from functools import partial
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt
import numpy as np
import numdifftools as nd


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

class Node(object):
    def __init__(self, loss_obj, parent=None, name="NoName", input_dim = 0):
        self.parent = parent
        self.name = name
        self.loss_obj = loss_obj
        self.w = np.zeros(input_dim)
    def predict(self, x):
        return Node._predict(self.w, x)
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
        pred = partial(self._predict, x=x)
        param_grad_func = nd.Derivative(pred)
        grad = param_grad_func(self.w)
        self.w = self.w - step_size*grad*loss
        return self.w
    @staticmethod
    def _predict(w, x):
        return w.dot(x*x)
        #return w.dot(x*x)


feature_dim = 1
def f(x):
    #return 0.25*x
    return 1e-1*x*x 

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
    # compute running average
    partial_sums = compute_running_average(child_nodes, predictions)
    top_loss = top_node.loss(partial_sums[-1], pt.y)
    return partial_sums, top_loss

def main(num_pts, num_children, learning_rate=1.5, rand_seed=0):
    top_node= Node(SqLoss, parent=None, name="root", input_dim = 0)
    child_nodes= [Node(SqLoss, parent=top_node, input_dim = feature_dim, name='Child {:d}'.format(i))\
            for i in xrange(num_children)]
    
    validation_set = [pt for pt in dataset(num_pts, seed=rand_seed+1)]

    npprint = partial(np.array_str, precision = 3)

    for i,pt in enumerate(dataset(num_pts, seed=rand_seed)):
        # Compute loss on Validation set
        val_losses = [predict_layer(val_pt, child_nodes, top_node)[1] for val_pt in validation_set]
        avg_val_loss = np.mean(val_losses)
        # Compute the partial sums, loss on current data point 
        partial_sums, top_loss = predict_layer(pt, child_nodes, top_node)
        learner_weights = np.array([node.w for node in child_nodes])
        # get the gradient of the top loss at each partial sum
        true_val = pt.y 
        offset_partials = np.zeros(partial_sums.shape) + np.NaN
        offset_partials[1:] = partial_sums[:-1]
        offset_partials[0] = 0
        dlosses = [node.dloss(pred_val, true_val) for pred_val,node in zip(partial_sums, child_nodes)]
        step_size = 1./np.power((i+1), learning_rate)
        learner_weights = np.array([node.grad_step(pt.x, loss, step_size)\
                for (node, loss) in zip(child_nodes, dlosses)])
        if i % ceil(num_pts*0.1) == 0 or i == num_pts-1:
            print('Iteration {:d}/{:d}: (x={:.2g},y={:.2g})'.format(i+1, num_pts, pt.x, pt.y))
            print(' Avg validation loss on pt: {:.4g}'.format(avg_val_loss))
            print('  Top layer loss on pt: {:.4g}'.format(top_loss))
            print('  Child learner weights: {}'.format(npprint(learner_weights.ravel())))
            print('  Partial sums: {}'.format(npprint(partial_sums)))
            print('  Took descent step of step size {:.4g}...'.format(step_size))

        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("1d Online Gradient Boosting")
    parser.add_argument('-n', '--num_pts', default=4, help='number of data points to run')
    parser.add_argument('-c', '--num_children', default=3, help='number of children learners')
    parser.add_argument('-l', '--learning_rate', default=3.0, help='eta in step_size=1/(i+1)^eta')
    parser.add_argument('-s', '--seed', default=0, help='random seed for data gen')
    args = parser.parse_args()
    num_pts     = int(args.num_pts)
    num_child   = int(args.num_children)
    rand_seed   = int(args.seed)
    learning_rate = float(args.learning_rate)
    main(num_pts, num_child, learning_rate, rand_seed)






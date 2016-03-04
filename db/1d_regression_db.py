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
        self.w = self.w - step_size*grad.dot(loss)
        return self.w
    @staticmethod
    def _predict(w, x):
        return w.dot(x)


feature_dim = 1
def f(x):
    return 0.25*x
    #return x*x 

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

def main():
    top_node= Node(SqLoss, parent=None, name="root", input_dim = 0)
    num_child = 3
    child_nodes= [Node(SqLoss, parent=top_node, input_dim = feature_dim, name='Child {:d}'.format(i))\
            for i in xrange(num_child)]
    
    num_pts =  50
    for i,pt in enumerate(dataset(num_pts, seed=1)):
        print('Iteration {}/{}: (x={:.4g},y={:.4g})'.format(i+1, num_pts, pt.x, pt.y))
        # predict up
        predictions = np.array([node.predict(pt.x) for node in child_nodes])
        # compute running average
        partial_sums = compute_running_average(child_nodes, predictions)
        top_loss = top_node.loss(partial_sums[-1], pt.y)
        print ' Top layer loss on pt: {:.4g}'.format(top_loss)
        learner_weights = np.array([node.w for node in child_nodes])
        print '  Child learner weights: {}'.format(learner_weights.ravel())
        print '  Partial sums: \t {}'.format(partial_sums)
        # get the gradient of the top loss at each partial sum
        true_val = pt.y 
        dlosses = [node.dloss(pred_val, true_val) for pred_val,node in zip(partial_sums, child_nodes)]
        step_size = 1./np.power((i+1), 1.1)
        learner_weights = np.array([node.grad_step(pt.x, loss, step_size)\
                for (node, loss) in zip(child_nodes, dlosses)])
        print ' Took descent step of step size {:.4g}...'.format(step_size)

        
    

if __name__ == "__main__":
    main()






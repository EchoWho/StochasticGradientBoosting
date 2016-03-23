#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from math import ceil, floor, sqrt, cos, sin
from timer import Timer

from collections import namedtuple
DataPt = namedtuple('DataPt', ['x','y'])

from IPython import embed

npprint = partial(np.array_str, precision = 3)

FEATURE_DIM=1
MAX_X=5
# dataset generator function
def dataset(num_pts, f, seed=None):
    if seed is not None:
      np.random.seed(seed)
    for _ in xrange(num_pts-1):
        x = -MAX_X+2.*MAX_X*np.random.rand(1)[0]
        y = f(x)
        yield DataPt(np.array([x]),y)
    yield DataPt(np.array([1]), f(1))


# from sgd_fast sklearn source
class SquaredLoss(object):
  """Squared loss traditional used in linear regression."""
  def loss(self, p, y):
    return 0.5 * (p - y) * (p - y)
  def dloss(self, p, y):
    """ p is the prediction, y is the true value"""
    return p - y

class LogisticLoss(object):
  def loss(self, p, y):
    return np.log(1 + np.exp(-y*p))
  def dloss(self, p, y):
    return (-y)/(1 + np.exp(y*p))

class ReLuMean(object):
  def mean(self, x):
      if x < 0:
          return 0.5*x
      return x
  def dmean(self, x): 
    if x < 0:
        return 0.5
    return 1.

class LinearMean(object):
  def mean(self, x):
    return x
  def dmean(self, x):
    return 1.

class SigmoidMean(object):
  def mean(self, x):
    return 1.0/(1+np.exp(-x))
  def dmean(self, x): 
    m = self.mean(x)
    return m*(1.0-m)

class LinearAddSigmoidMean(object):
  def __init__(self):
    self.lm = LinearMean()
    self.sm = SigmoidMean()
  def mean(self, x):
    return self.lm.mean(x) + self.sm.mean(x)
  def dmean(self, x):
    return self.lm.dmean(x) + self.sm.dmean(x)

# original signed square is bad because 0 gradient at 0;
# this signed square is still bad because exploding gradient.
# Mean function must have bounded gradients. 
class SignedSquareMean(object):
  def mean(self, x):
    if x >= 0:
      return 0.5*((x+1.0)**2 - 1.0)
    return -0.5*((x-1.0)**2 - 1.0)
  def dmean(self, x):
    if x >= 0:
      return x+1.0
    return 1.0-x

class GatedSignedSquareRootMean(object):
  def mean(self, x):
    absx = np.abs(x)
    if absx < 0.25:
      return 2.*x
    return np.sqrt(absx)
  def dmean(self, x):
    absx = np.abs(x)
    if absx < 0.25:
      return 2.
    return 0.5 / np.sqrt(absx)

class BoostNode(object):
  def __init__(self, name, loss_obj, mean_func, children=[], children_indices=[]):
    self.name = name
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.children = children
    self.children_indices = children_indices
    self.n_children = len(self.children_indices)
    self.weights = np.zeros(self.n_children) #np.ones(self.n_children)/self.n_children
    self.b = 0.0

  def compute_partial_sum(self, children_pred):
    psum = [self.b]
    for i, ci in enumerate(self.children_indices):
      psum.append(psum[-1] + self.weights[i]*children_pred[ci])
    return psum 

  def predict(self, children_pred):
    psum = self.compute_partial_sum(children_pred)
    return self.mean_func.mean(psum[-1])
    
  def get_targets_and_update(self, children_pred, y, step_size=1.0):
    # prediction
    psum = self.compute_partial_sum(children_pred)
    yps = [ self.mean_func.mean(ps) for ps in psum ]
    # children targets for continued learning
    yts = [ -self.loss_obj.dloss(yps[i], y) * self.mean_func.dmean(psum[i]) \
        for i in range(self.n_children+1) ]

    #print 'y   : {}'.format(npprint(y))
    #print 'psum: {}'.format(npprint(np.array(psum)))
    #print 'yps : {}'.format(npprint(np.array(yps)))
    #print 'yts : {}'.format(npprint(np.array(yts).ravel()))
    #print 'wgts: {}'.format(npprint(self.weights.ravel()))

    assert( not np.isinf(yts[-1]))
    assert( not np.isnan(yts[-1]))

    self.b += yts[0][0] * step_size
    for i in range(self.n_children):
      #sgn_tgt = yts[i] > -1e-16
      #sgn_pred = children_pred[self.children_indices[i]] > -1e-16
      #self.weights[i] -= (np.float64(sgn_tgt == sgn_pred) -0.5) * step_size
      self.weights[i] += yts[i+1] * children_pred[self.children_indices[i]] * step_size
      #self.weights[i] -= yts[-1] * children_pred[self.children_indices[i]] * step_size
      #grad = yts[-1] * children_pred[self.children_indices[i]] 
      #grad_sign = np.sign(grad)[0]
      #self.weights[i] -= grad_sign * step_size
      #self.weights[i] = 2./(i+1.)

    return self.children_indices, yts[:-1]

class RegressionNode(object):
  def __init__(self, name, loss_obj, mean_func, input_dim):
    self.name = name
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.w = np.zeros(input_dim)
    self.b = 0
  
  def predict_linear(self, x):
    return np.dot(self.w, x)+self.b

  def predict(self, x):
    return self.mean_func.mean(self.predict_linear(x))

  def learn(self, x, y, step_size=1.0):
    pl = self.predict_linear(x)
    p = self.mean_func.mean(pl)
    grad = step_size * self.loss_obj.dloss(p, y) * self.mean_func.dmean(pl)
    self.w -= grad*x
    self.b -= grad

# This regression predicts 1 or -1. 
# The input target is y. The sign of y is the label.
# The magnitude of y is the weight. 
class RegressionNodeWithClassification(object):
  def __init__(self, name, loss_obj, mean_func, input_dim):
    self.name = name
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.w = np.zeros(input_dim)
    self.b = 0
  
  def predict_linear(self, x):
    return np.dot(self.w, x)+self.b

  def predict_prob(self, x):
    return self.mean_func.mean(self.predict_linear(x))

  def predict(self, x):
    return 2*(self.predict_prob(x)-0.5)

  def learn(self, x, y, step_size=1.0):
    pl = self.predict_linear(x)
    pp = self.mean_func.mean(pl)
    y_sgn = np.sign(y)
    y_wgt = np.abs(y)
    grad = step_size * self.loss_obj.dloss(pp, y_sgn) * self.mean_func.dmean(pl) * y_wgt
    self.w -= grad*x
    self.b -= grad


class DeepBoostGraph(object):
  def __init__(self, n_lvls, n_nodes, input_dim, loss_obj, mean_func):
    assert n_lvls == len(n_nodes)
    self.n_lvls = n_lvls
    self.n_nodes = n_nodes
    self.input_dim = input_dim
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.nodes = []
    for i in range(n_lvls):
      if i == 0:
        lvl_nodes = [RegressionNodeWithClassification('n_{}_{}'.format(i, ni), \
          loss_obj[i](), mean_func[i](), input_dim) for ni in range(n_nodes[i])]
      else:
        children_indices = np.reshape(np.arange(n_nodes[i-1]), (n_nodes[i], n_nodes[i-1]/n_nodes[i]))
        lvl_nodes = [BoostNode('n_{}_{}'.format(i, ni), loss_obj[i](), mean_func[i](), self.nodes[-1], \
          children_indices[ni].ravel()) for ni in range(n_nodes[i])]
      self.nodes.append(lvl_nodes)

  def full_pred(self, x):
    children_preds = []
    for i in range(self.n_lvls):
      if i==0:
        children_pred = [node.predict(x) for node in self.nodes[0]]
      else:
        children_pred = [node.predict(children_pred) for node in self.nodes[i]] 
      children_preds.append(children_pred)
    return children_preds

  def predict(self, x):
    children_pred = [node.predict(x) for node in self.nodes[0]]
    for i in range(1, self.n_lvls):
      children_pred = [node.predict(children_pred) for node in self.nodes[i]] 
    return children_pred[0]

  def predict_and_learn(self, x, y, boost_step_size=1e-3, regress_step_size=1.0):
    children_preds = self.full_pred(x)
    ys = [y]
    for i in reversed(range(self.n_lvls)):
      if i>0:
        ys_i = []
        for (ni, node) in enumerate(self.nodes[i]):
          child_idx, ys_ni = node.get_targets_and_update(children_preds[i-1], ys[ni], boost_step_size) 
          ys_i = ys_i + ys_ni
        ys = ys_i
      else:
        for (ni, node) in enumerate(self.nodes[i]):
          node.learn(x, ys[ni], regress_step_size)
          
        
        ws = np.array([node.w for node in self.nodes[i]]).ravel()
        bs = np.array([node.b for node in self.nodes[i]]).ravel()
#        print 'w   : {}'.format(npprint(ws))
#        print 'b   : {}'.format(npprint(bs))

    return children_preds[-1][0]

def main():
  #n_nodes = [9**i for i in reversed(range(n_lvls))]
  #n_nodes = [100, 20, 1]
  n_nodes = [30, 1]
  n_lvls =  len(n_nodes)
  sq_loss = SquaredLoss()
  loss_obj = [LogisticLoss for _ in xrange(n_lvls-1)]
  mean_func = [SigmoidMean for _ in xrange(n_lvls-1)]
  loss_obj.append(SquaredLoss)
  mean_func.append(LinearMean)

  input_dim = FEATURE_DIM
  
  dbg = DeepBoostGraph(n_lvls, n_nodes, input_dim, loss_obj, mean_func)

  #f = lambda x : mean_func[0].mean(np.dot(np.array([1.0]), x))
  #fsig = lambda x:mean_func[0].mean(x)
  #f = lambda x: \
  #  fsig(1.3*fsig(fsig(np.array([x+1]))*2.0 + fsig(np.array([x-1]))*0.5) - \
  #       0.7*fsig(fsig(np.array([x+2]))*5.2 + fsig(np.array([x-3]))*3.1))
  #f = lambda x : np.array([np.cos(x+1) + x*np.sin(x-1) + x])
  f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])
  #f = lambda x : np.array([8.*x + 2.8])

  train_set = [pt for pt in dataset(5001, f, 91612)]
  val_set = [pt for pt in dataset(201, f)]
  val_set = sorted(val_set, key = lambda x: x.x)

  max_epoch = 20
  t=0
  boost_lr = 1e-3
  regress_lr = 5e-3
  for epoch in range(max_epoch):
    for (si, pt) in enumerate(train_set):
      t+=1
      #dbg.predict_and_learn(pt.x, pt.y, 5e-1/np.power(t,3.0), 1e-1/np.power(t,1.0))
      #dbg.predict_and_learn(pt.x, pt.y, 1e-2/np.power(t_step+1,0.25), 5e-3/np.power(t_step+1,0.25))
      dbg.predict_and_learn(pt.x, pt.y, boost_lr, regress_lr)

      if si%1000==0:
        avg_loss = 0
        for (vi, vpt) in enumerate(val_set):
          p = dbg.predict(vpt.x)
          avg_loss += sq_loss.loss(p, vpt.y)
        avg_loss /= np.float64(len(val_set))
        
        print_x = [-4,-3,-2,-1,0,1,2,3,4]
        print 'Prediction on {}: \n {} \n {}'.format(print_x,\
          npprint(np.array([f(prx) for prx in print_x]).ravel()), \
          npprint(np.array([dbg.predict(prx) for prx in print_x]).ravel()))
        print 'Avg Loss at t={} is: {}'.format(t, avg_loss)
    #boost_lr *= 0.85
    #regress_lr *= 0.85

  print_info = [ (dbg.predict(pt.x), pt.y, pt.x) for pt in val_set ]
  y_preds, y_gt, x_gt = zip(*print_info)
  plt.plot(x_gt, y_preds, label='Predictions')
  plt.plot(x_gt, y_gt, label='Ground Truth')
  plt.legend(loc=4)
  plt.show(block=False)
  embed()

if __name__ == "__main__":
  main()

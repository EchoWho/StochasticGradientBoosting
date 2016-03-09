#!/usr/bin/env python

import numpy as np
from functools import partial
from math import ceil, floor, sqrt, cos, sin
from timer import Timer

from collections import namedtuple
DataPt = namedtuple('DataPt', ['x','y'])

npprint = partial(np.array_str, precision = 3)

FEATURE_DIM=1
MAX_X=5
# dataset generator function
def dataset(num_pts, f, seed=None):
    if seed is not None:
      np.random.seed(seed)
    for _ in xrange(num_pts):
        x = -MAX_X+2.*MAX_X*np.random.rand(1)[0]
        y = f(x)
        yield DataPt(np.array([x]),y)


# from sgd_fast sklearn source
class SquaredLoss(object):
  """Squared loss traditional used in linear regression."""
  def loss(self, p, y):
    return 0.5 * (p - y) * (p - y)
  def dloss(self, p, y):
    """ p is the prediction, y is the true value"""
    return p - y

class LinearMean(object):
  def mean(self, x):
    return x
  def dmean(self, x):
    return 1.

class SignedSquareMean(object):
  def mean(self, x):
    return x*x* np.float64((x>0) - 0.5)
  def dmean(self, x):
    return np.abs(x)

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

  def compute_partial_sum(self, children_pred):
    psum = [0.]
    for i, ci in enumerate(self.children_indices):
      #wgt = 2./(i+2.)
      #psum.append(psum[-1]*(1.-wgt) + wgt*children_pred[ci])
      wgt = 2./(i+2.)
      psum.append(psum[-1] + wgt*children_pred[ci])
    return psum 

  def predict(self, children_pred):
    psum = self.compute_partial_sum(children_pred)
    return psum[-1]
    
  def compute_ys(self, children_pred, y):
    psum = self.compute_partial_sum(children_pred)
    ci_ys = [ (ci, -self.loss_obj.dloss(psum[i], y) * self.mean_func.dmean(psum[i])) \
        for i, ci in enumerate(self.children_indices) ]
    print 'y   : {}'.format(npprint(y))
    print 'psum: {}'.format(npprint(np.array(psum)))
    print 'ys:   {}'.format(npprint(np.array([y[0] for (ci, y) in ci_ys])))
    return ci_ys
  
class RegressionNode(object):
  def __init__(self, name, loss_obj, mean_func, input_dim):
    self.name = name
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.w = np.zeros(input_dim)

  def predict(self, x):
    return self.mean_func.mean(np.dot(self.w, x))

  def learn(self, x, y, step_size=1.0, p=None):
    if p is None:
      p = self.predict(x)
    self.w -= step_size * self.loss_obj.dloss(p, y) * self.mean_func.dmean(p) * x
  
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
        lvl_nodes = [RegressionNode('n_{}_{}'.format(i, ni), loss_obj[i], mean_func[i], input_dim)  \
          for ni in range(n_nodes[i])]
      else:
        children_indices = np.reshape(np.arange(n_nodes[i-1]), (n_nodes[i], n_nodes[i-1]/n_nodes[i]))
        lvl_nodes = [BoostNode('n_{}_{}'.format(i, ni), loss_obj[i], mean_func[i], self.nodes[-1], \
          children_indices[ni]) for ni in range(n_nodes[i])]
      self.nodes.append(lvl_nodes)

  def predict(self, x):
    children_pred = [node.predict(x) for (ni, node) in enumerate(self.nodes[0])]
    for i in range(1, self.n_lvls):
      children_pred = [node.predict(children_pred) for (ni, node) in enumerate(self.nodes[i])] 
    return children_pred[0]

  def predict_and_learn(self, x, y, step_size=1.0):
    children_preds = []
    for i in range(self.n_lvls):
      if i==0:
        children_pred = [node.predict(x) for (ni, node) in enumerate(self.nodes[0])]
      else:
        children_pred = [node.predict(children_pred) for (ni, node) in enumerate(self.nodes[i])] 
      children_preds.append(children_pred)

    ys = [y]
    for i in reversed(range(self.n_lvls)):
      if i>0:
        ys_i = []
        for (ni, node) in enumerate(self.nodes[i]):
          print "x   : {}".format(x)
          ci_ys = node.compute_ys(children_preds[i-1], ys[ni])
          ys_ni = [ y  for (ci, y) in ci_ys ] 
          ys_i = ys_i + ys_ni
        ys = ys_i
      else:
        for (ni, node) in enumerate(self.nodes[i]):
          node.learn(x, ys[ni], step_size, p=children_preds[0][ni])
          
        
        ws = np.array([node.w for node in self.nodes[i]]).ravel()
        print 'w   : {}'.format(npprint(ws))

    return children_preds[-1][0]

def main():
  n_lvls = 2
  n_nodes = [9**i for i in reversed(range(n_lvls))]
  sq_loss = SquaredLoss()
  loss_obj = [SquaredLoss() for i in range(n_lvls)]
#  mean_func = [SignedSquareMean(), GatedSignedSquareRootMean(), SignedSquareMean()]
  mean_func = [LinearMean(), LinearMean()]
  input_dim = FEATURE_DIM
  
  dbg = DeepBoostGraph(n_lvls, n_nodes, input_dim, loss_obj, mean_func)

  f = lambda x : np.dot(np.array([1.0]), x)
  
  train_set = [pt for pt in dataset(200, f, 91612)]
  val_set = [pt for pt in dataset(200, f)]

  max_epoch = 1
  t=0
  for epoch in range(max_epoch):
    for (si, pt) in enumerate(train_set):
      t+=1
      dbg.predict_and_learn(pt.x, pt.y, 1.1/np.power(t+1,0.5))

      if si%10==0:
        avg_loss = 0
        for (vi, vpt) in enumerate(val_set):
          p = dbg.predict(vpt.x)
          avg_loss += sq_loss.loss(p, vpt.y)
        avg_loss /= np.float64(len(val_set))
        
        print 'Prediction on f(1): {}, {}'.format(f(1), dbg.predict(1))
        print 'Avg Loss at t={} is: {}'.format(t, avg_loss)

if __name__ == "__main__":
  main()

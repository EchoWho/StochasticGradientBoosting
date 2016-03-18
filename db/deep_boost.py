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
    self.n_children = len(self.children_indices)
    self.eta = 2.0/self.n_children
    self.sigma = np.zeros(self.n_children)

  def compute_partial_sum(self, children_pred):
    psum = [0.]
    for i, ci in enumerate(self.children_indices):
      ## Algo 2. of Online Gradient Boosting.
      #wgt = 2./(i+2.)
      #psum.append(psum[-1]*(1.-wgt) + wgt*children_pred[ci])
      
      ## No shrinking of the old
      #wgt = 2./(i+2.)
      #psum.append(psum[-1] + wgt*children_pred[ci])

      ## Algo 1. of Online Gradient Boositng
      psum.append(psum[-1]*(1-self.sigma[i]*self.eta) + self.eta*children_pred[ci])
    return psum 

  def predict(self, children_pred):
    psum = self.compute_partial_sum(children_pred)
    return psum[-1]
    
  def learn(self, children_pred, y, step_size=1.0):
    psum = self.compute_partial_sum(children_pred)
    ys = [ -self.loss_obj.dloss(psum[i], y) * self.mean_func.dmean(psum[i]) \
        for i, ci in enumerate(self.children_indices) ]

    self.sigma += step_size*np.array(psum[0:-1])*np.array(ys).ravel()
    self.sigma = np.maximum(np.minimum(self.sigma, 1), 0)

    print 'y   : {}'.format(npprint(y))
    print 'psum: {}'.format(npprint(np.array(psum)))
    print 'ys:   {}'.format(npprint(np.array([y[0] for y in ys])))
    print 'sig : {}'.format(npprint(self.sigma))
    return self.children_indices, ys
  
class RegressionNode(object):
  def __init__(self, name, loss_obj, mean_func, input_dim):
    self.name = name
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.w = np.zeros(input_dim)
    self.b = 0

  def predict(self, x):
    return self.mean_func.mean(np.dot(self.w, x)+self.b)

  def learn(self, x, y, step_size=1.0, p=None):
    if p is None:
      p = self.predict(x)
    grad = step_size * self.loss_obj.dloss(p, y) * self.mean_func.dmean(p)
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

  def predict_and_learn(self, x, y, step_size=1.0, boost_step_scale=1e-2, 
        regress_step_scale=1.0):
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
          child_idx, ys_ni = node.learn(children_preds[i-1], ys[ni], 
              boost_step_scale*step_size)
          ys_i = ys_i + ys_ni
        ys = ys_i
      else:
        for (ni, node) in enumerate(self.nodes[i]):
          node.learn(x, ys[ni], regress_step_scale*step_size, p=children_preds[0][ni])
          
        
        ws = np.array([node.w for node in self.nodes[i]]).ravel()
        print 'w   : {}'.format(npprint(ws))
        bs = np.array([node.b for node in self.nodes[i]]).ravel()
        print 'b   : {}'.format(npprint(bs))

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

  f = lambda x : mean_func[0].mean(np.dot(np.array([1.0]), x))
  
  train_set = [pt for pt in dataset(500, f, 91612)]
  val_set = [pt for pt in dataset(200, f)]

  max_epoch = 1
  t=0
  for epoch in range(max_epoch):
    for (si, pt) in enumerate(train_set):
      t+=1
      dbg.predict_and_learn(pt.x, pt.y, 1.0/np.power(t+1,0.5))

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

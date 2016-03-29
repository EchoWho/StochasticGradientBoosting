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

# Target y are in {-1, 1}, used for sigmoidClassification
class LogisticLoss(object):
  def loss(self, p, y):
    return np.log(1 + np.exp(-y*p))
  def dloss(self, p, y):
    return (-y)/(1 + np.exp(y*p))

# Target y is in {-1, 1}. p is in \R
class HedgeLoss(object):
  def loss(self, p, y):
    return np.maximum(1.-p*y, 0)
  def dloss(self, p, y):
    l = self.loss(p,y)
    if l > 0:
      return -y
    return 0

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

class SigmoidClassifierMean(object):
  def __init__(self):
    self.sm = SigmoidMean()
  def mean(self, x):
    return 2*(self.sm.mean(x)-0.5)
  def dmean(self, x):
    return 2*self.sm.dmean(x)

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


# Optimizer classes: take in a gradient $g$, and returns the value to subtract from the 
# the parameters: i.e., x' = x - \eta * opt.update(g), where \eta is a step_size. 
class SGDOptimizer(object):
  def __init__(self, input_dim):
    self.dim = input_dim   
  
  def update(self, grad):
    return grad

class ADAMOptimizer(object):
  def __init__(self, input_dim, b1, b2, eps):
    assert(b1 < 1. and b1 > 0.)
    assert(b2 < 1. and b2 > 0.)
    assert(eps >= 0.)
    self.dim = input_dim
    self.t = 1
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.b1_pow_t = self.b1
    self.b2_pow_t = self.b2
    self.m = np.zeros(input_dim)
    self.v = np.zeros(input_dim)

  def update(self, g):
    # update momentum
    self.m = self.b1 * self.m + (1. - self.b1) * g
    self.v = self.b2 * self.v + (1. - self.b2) * g**2
    # compute unbiased momenetum
    m = self.m / (1. - self.b1_pow_t)
    v = self.v / (1. - self.b2_pow_t)
    # update value related to timestamp t
    self.b1_pow_t *= self.b1
    self.b2_pow_t *= self.b2
    self.t += 1
    return m / (self.eps + np.sqrt(v))

class BoostNode(object):
  def __init__(self, name, loss_obj, mean_func, classification, \
               children, children_indices, optimizer):
    self.name = name
    self.classification = classification
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.children = children
    self.children_indices = children_indices
    self.n_children = len(self.children_indices)
    self.weights = np.zeros(self.n_children) #np.ones(self.n_children)/self.n_children
    self.b = 0.0
    assert(optimizer.dim == self.n_children + 1)
    self.optimizer = optimizer

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
    wgt = 1.0
    if (self.classification):
      wgt = np.abs(y)
      y = np.sign(y)
    yts = [ -self.loss_obj.dloss(yps[i], y) * self.mean_func.dmean(psum[i]) * wgt \
        for i in range(self.n_children+1) ]

    #print 'y   : {}'.format(npprint(y))
    #print 'psum: {}'.format(npprint(np.array(psum)))
    #print 'yps : {}'.format(npprint(np.array(yps)))
    #print 'yts : {}'.format(npprint(np.array(yts).ravel()))
    #print 'wgts: {}'.format(npprint(self.weights.ravel()))

    # prevent explotion by failing
    assert( not np.isinf(yts[-1]))
    assert( not np.isnan(yts[-1]))
    # compute gradient
    grad = np.zeros(1+self.n_children)
    grad[0] = -yts[0][0]
    grad[1:] = [ -yts[i+1] * children_pred[self.children_indices[i]] \
      for i in range(self.n_children)]
    # update with optimized update
    opt_update = self.optimizer.update(grad) * step_size
    self.b -= opt_update[0]
    self.weights -= opt_update[1:] 
    return self.children_indices, yts[:-1]

class LeafNode(object):
  def __init__(self, name, loss_obj, mean_func, input_dim, classification, optimizer):
    self.name = name
    self.classification = classification
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.w = np.zeros(input_dim)
    self.b = 0
    assert(optimizer.dim == input_dim + 1)
    self.optimizer = optimizer
  
  def predict_linear(self, x):
    return np.dot(self.w, x)+self.b

  def predict(self, x):
    return self.mean_func.mean(self.predict_linear(x))

  def learn(self, x, y, step_size=1.0):
    pl = self.predict_linear(x)
    p = self.mean_func.mean(pl)
    wgt = 1.0
    if (self.classification): 
      wgt = np.abs(y)
      y = np.sign(y)
    # gradient with respect to the linear prediction wx+b
    grad_l = self.loss_obj.dloss(p, y) * self.mean_func.dmean(pl) * wgt
    # full gradient 
    grad = np.zeros(1 + self.w.shape[0])
    grad[0] = grad_l
    grad[1:] = grad_l*x
    # optimized update
    opt_update = self.optimizer.update(grad) * step_size
    self.b -= opt_update[0]
    self.w -= opt_update[1:]

class DeepBoostGraph(object):
  def __init__(self, n_lvls, n_nodes, input_dim, loss_obj, mean_func, adam_b1, adam_b2, adam_eps):
    assert n_lvls == len(n_nodes)
    self.n_lvls = n_lvls
    self.n_nodes = n_nodes
    self.input_dim = input_dim
    self.loss_obj = loss_obj
    self.mean_func = mean_func
    self.nodes = []
    for i in range(n_lvls):
      if i == 0:
        lvl_nodes = [LeafNode('n_{}_{}'.format(i, ni), \
          loss_obj[i](), mean_func[i](), input_dim, True, \
          #SGDOptimizer(1+input_dim)) \
          ADAMOptimizer(1+input_dim, adam_b1, adam_b2, adam_eps))
          for ni in range(n_nodes[i])]
      else:
        children_indices = np.reshape(np.arange(n_nodes[i-1]), (n_nodes[i], n_nodes[i-1]/n_nodes[i]))
        lvl_nodes = [BoostNode('n_{}_{}'.format(i, ni), loss_obj[i](), \
          mean_func[i](), i < n_lvls-1, \
          self.nodes[-1], children_indices[ni].ravel(), \
          #SGDOptimizer(1+children_indices[ni].ravel().shape[0])) 
          ADAMOptimizer(1+children_indices[ni].ravel().shape[0], adam_b1, adam_b2, adam_eps)) 
          for ni in range(n_nodes[i])]
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
  n_nodes = [50, 25, 1]
  n_lvls =  len(n_nodes)
  sq_loss = SquaredLoss()
  loss_obj = [HedgeLoss for _ in xrange(n_lvls-1)]
  mean_func = [SigmoidClassifierMean for _ in xrange(n_lvls-1)]
  loss_obj.append(SquaredLoss)
  mean_func.append(LinearMean)

  input_dim = FEATURE_DIM
  
  adam_b1 = 0.9
  adam_b2 = 0.9
  adam_eps = 1e-5
  boost_lr = 5e-3
  regress_lr = 8e-3
  #lr_gamma = 1
  dbg = DeepBoostGraph(n_lvls, n_nodes, input_dim, loss_obj, mean_func, adam_b1, adam_b2, adam_eps)

  f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])

  train_set = [pt for pt in dataset(5001, f, 9122)]
  val_set = [pt for pt in dataset(201, f)]
  val_set = sorted(val_set, key = lambda x: x.x)

  max_epoch = 30
  t=0
  for epoch in range(max_epoch):
    np.random.shuffle(train_set)
    for (si, pt) in enumerate(train_set):
      t+=1
      dbg.predict_and_learn(pt.x, pt.y, boost_lr, regress_lr)

      #if si == len(train_set)-1:
      if (si <= 4000 and si>=1000 and si%1000 == 0) or (si == len(train_set) -1):
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
    #boost_lr *= lr_gamma
    #regress_lr *= lr_gamma

  print_info = [ (dbg.predict(pt.x), pt.y, pt.x) for pt in val_set ]
  y_preds, y_gt, x_gt = zip(*print_info)
  plt.plot(x_gt, y_preds, label='Predictions')
  plt.plot(x_gt, y_gt, label='Ground Truth')
  plt.legend(loc=4)
  plt.show(block=False)
  embed()

if __name__ == "__main__":
  main()

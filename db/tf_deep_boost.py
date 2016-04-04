#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import itertools
from functools import partial
from math import ceil, floor, sqrt, cos, sin
from timer import Timer
from collections import namedtuple
from IPython import embed

import tensorflow as tf

DataPt = namedtuple('DataPt', ['x','y'])
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

flags = tf.app.flags
FLAGS = flags.FLAGS

def sigmoid_clf_mean(x):
  return tf.sub(tf.scalar_mul(2.0, tf.sigmoid(x)), tf.ones_like(x))

def logistic_loss_eltws(yp, y):
  return tf.log(tf.add(tf.exp(tf.neg(tf.mul(yp, y))), tf.ones_like(y)))

def square_loss_eltws(yp, y):
  return tf.scalar_mul(0.5, tf.square(tf.sub(yp, y)))

class TFLeafNode(object):
  """ Apply a K-dim generalized linear model 
      that maps the input x to K independently mapped dims.
  """

  def __init__(self, name, dim, mean_type, loss_type, opt_type):
    """ record params of the node for construction in the future.

    Args:
      dim: (D, K)
    """
    self.name = name
    self.dim = dim
    self.mean_type = mean_type #e.g., tf.nn.relu
    #self.mean_type = lambda x, n : tf.nn.bias_add(tf.scalar_mul(2.0, tf.sigmoid(x)), -1.0, n)
    self.loss_type = loss_type # element-wise loss 
    self.opt_type = opt_type   #e.g., tf.train.GradientDescentOptimizer

  def inference(self, x):
    """ Construct inference part of the node. 
      
    Args:
      x: input tensor, float - [batch_size, dim[0]]
    """
    # linear transformation
    with tf.name_scope(self.name):
      self.w = tf.Variable(
          tf.truncated_normal([self.dim[0], self.dim[1]],
                              stddev=1.0 / sqrt(float(self.dim[0]))),
          name='weights')
      self.b = tf.Variable(tf.zeros([self.dim[1]]), name='biases')
      self.pred = self.mean_type(tf.nn.bias_add(tf.matmul(x, self.w), self.b))
    return self.pred
      
  def loss(self, y):
    """ Construct TF loss graph given the inference graph.

    Args:
      y: target (sign and magnitude on each dim), float - [batch_size, dim[1]] 
    """
    with tf.name_scope(self.name):
      y_sgn = tf.sign(y, name='y_sgn') # target of dim
      y_abs = tf.abs(y, name='y_abs')  # wgt of each dim 

      # weighted average of loss_type(pred, y_sgn)
      ind_loss = self.loss_type(self.pred, y_sgn)
      wgt_loss = tf.mul(y_abs, ind_loss, 'wgt_loss')
      self.loss = tf.reduce_mean(wgt_loss, name='avg_loss')
    return self.loss

  def training(self, lr):
    self.optimizer = self.opt_type(lr)
    self.train_op = self.optimizer.minimize(self.loss)
    return self.train_op

class TFBoostNode(object):
  def __init__(self, name, dim, mean_type, loss_type, opt_type, convert_y=True):
    """ record params of the node for construction in the future.

    Args:
      dim: (D, K)
    """
    self.name = name
    self.dim = dim
    self.mean_type = mean_type #e.g., tf.nn.relu
    self.loss_type = loss_type # element-wise loss 
    self.opt_type = opt_type   #e.g., tf.train.GradientDescentOptimizer
    self.convert_y = convert_y

  def inference(self, children_preds):
    self.n_children = len(children_preds)
    with tf.name_scope(self.name):
      self.ps_b = tf.Variable(tf.zeros([1,self.dim[0]]), name='ps_bias')
      self.ps_ws = [tf.Variable(tf.zeros([1]), name='ps_weight_'+str(ci)) 
          for ci in range(self.n_children)]
      self.tf_w = tf.Variable(
          tf.truncated_normal([self.dim[0], self.dim[1]],
                              stddev=1.0 / sqrt(float(self.dim[0]))),
          name='tf_weights')
      
      # TODO test version that doesn't store all partial sums
      self.psums = []
      self.y_hats = []
      for i in range(self.n_children+1):
        if i == 0:
          ps = self.ps_b
        else:
          ps = tf.add(ps, tf.mul(children_preds[i-1], self.ps_ws[ci]))
        self.psums.append(ps)
        self.y_hats.append(self.mean_type(tf.matmul(ps, self.tf_w)))
    return self.y_hats[-1]

  def loss(self, y):
    with tf.name_scope(self.name):
      if self.convert_y:
        y_sgn = tf.sign(y, name='y_sgn')
        y_abs = tf.abs(y, name='y_abs')
        self.losses = [ tf.reduce_mean(tf.mul(y_abs, self.loss_type(y_hat, y_sgn)))
            for y_hat in self.y_hats ]
      else:
        self.losses = [ tf.reduce_mean(self.loss_type(y_hat, y)) for y_hat in self.y_hats ]
    return self.losses
      
  def training(self, lr):
    with tf.name_scope(self.name):
      # optimizer (n_children) is for bias
      self.train_ops = []
      self.children_tgts = []
      for i in range(self.n_children+1):
        opt = self.opt_type(lr)        
        if i == 0:
          train_op = opt.minimize(self.losses[i], var_list=[self.tf_w, self.ps_b])
        else:
          train_op = opt.minimize(self.losses[i], var_list=[self.tf_w, self.ps_ws[i-1]])
          grad_ps = tf.neg(tf.gradients(self.losses[i], [self.psums[i-1]])[0])
          self.children_tgts.append(grad_ps)
        self.train_ops.append(train_op)
      #endfor 
    return self.train_ops, self.children_tgts

class TFDeepBoostGraph(object):
  def __init__(self, dims, n_nodes, mean_types, loss_types, opt_types, lr_boost, lr_leaf):
    # dims: [input_d, out_d0, out_d1, ..., out_D==output_d]
    self.dims = dims
    self.n_nodes = n_nodes
    self.mean_types = mean_types
    self.loss_types = loss_types
    self.opt_types = opt_types
    self.lr_boost = lr_boost
    self.lr_leaf = lr_leaf

    # construct inference from bottom up
    print 'Construct inference()'
    self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]))
    self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]))
    self.ll_nodes = []
    for i in range(len(n_nodes)):
      dim = (dims[i], dims[i+1])
      if i == 0:
        l_nodes = [TFLeafNode('leaf'+str(ni), dim, mean_types[i], loss_types[i], 
            opt_types[i]) for ni in range(n_nodes[i])]  
        l_preds = map(lambda node : node.inference(self.x_placeholder), l_nodes)
      else:
        l_nodes = [TFBoostNode('boost'+str(ni), dim, mean_types[i], loss_types[i], 
            opt_types[i], i<len(n_nodes)-1) for ni in range(n_nodes[i])]  

        assert(n_nodes[i-1] % n_nodes[i] == 0)
        nc = n_nodes[i-1] / n_nodes[i] #n_children
        l_preds = map(lambda i : l_nodes[i].inference(l_preds[i*nc:(i+1)*nc]), range(n_nodes[i]))
      self.ll_nodes.append(l_nodes)
    #endfor
    self.pred = l_preds[0] # dbg.inference()

    print 'Construct loss() and training()'
    # construct loss and training_op from top down
    tgts = [self.y_placeholder] # prediction target of nodes on a level (back to front)
    ll_train_ops = []
    for i in reversed(range(len(n_nodes))):
      print 'depth={}'.format(i)
      l_nodes = self.ll_nodes[i]
      _ =  map(lambda ni : l_nodes[ni].loss(tgts[ni]), range(n_nodes[i])) 
      if i > 0:
        trainop_tgts = map(lambda nd : nd.training(lr_boost), l_nodes)
        l_train_ops, tgts = zip(*trainop_tgts)
        l_train_ops = list(itertools.chain.from_iterable(l_train_ops)) #flatten the list
        tgts = list(itertools.chain.from_iterable(tgts))
      else:
        l_train_ops = map(lambda x : x.training(lr_leaf), l_nodes) #every leaf node has one op.
        
      ll_train_ops.append(l_train_ops)
    #endfor

    print 'done.'
    #flatten all train_ops in one list
    self.train_ops = list(itertools.chain.from_iterable(ll_train_ops)) # dbg.training()
  #end __init__

  def fill_feed_dict(self, x, y):
    return { self.x_placeholder: x, self.y_placeholder: y }

  def inference(self):
    return self.pred

  def training(self):
    return self.train_ops

  def evaluation(self):
    return self.ll_nodes[-1][0].losses[-1]

def main(_):
  n_nodes = [40, 20, 1]
  n_lvls = len(n_nodes)
  mean_types = [ sigmoid_clf_mean for lvl in range(n_lvls-1) ]
  mean_types.append(lambda x : x)
  loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
  loss_types.append(square_loss_eltws)
  opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]

  input_dim = 1
  output_dim = 1
  dims = [ input_dim, 1, 1, output_dim ] 

  lr_boost = 5e-4
  lr_leaf = 5e-4

  # modify the default tensorflow graph.
  dbg = TFDeepBoostGraph(dims, n_nodes, mean_types, loss_types, opt_types, lr_boost, lr_leaf)

 
  f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])

  train_set = [pt for pt in dataset(5000, f, 9122)]
  val_set = [pt for pt in dataset(201, f)]
  val_set = sorted(val_set, key = lambda x: x.x)

  x_val = [ pt.x for pt in val_set]
  y_val = [ pt.y for pt in val_set]

  init = tf.initialize_all_variables()
  sess = tf.Session()
  print 'Initializing...'
  sess.run(init)
  print 'Initialization done'
  
  t = 0
  batch_size = 5
  val_interval = 1000
  max_epoch = 15
  for epoch in range(max_epoch):
    np.random.shuffle(train_set)
    for si in range(0, len(train_set), batch_size):
      si_end = min(si+batch_size, len(train_set))
      x = [ pt.x for pt in train_set[si:si_end] ]
      y = [ pt.y for pt in train_set[si:si_end] ]
      #print 'train epoch={}, start={}'.format(epoch, si)
      sess.run(dbg.training(), feed_dict=dbg.fill_feed_dict(x, y))

      
      if t % val_interval == 0:
        preds, avg_loss = sess.run([dbg.inference(), dbg.evaluation()], 
                                   feed_dict=dbg.fill_feed_dict(x_val, y_val))
        assert(not np.isnan(avg_loss))
        print 't={} avg_loss : {}'.format(t, avg_loss)
      t += si_end-si

  
  plt.plot(x_val, preds, label='Prediction')
  plt.plot(x_val, y_val, label='Ground Truth')
  plt.legend(loc=4)
  plt.show(block=False)
  embed()

if __name__ == '__main__':
  main(0)

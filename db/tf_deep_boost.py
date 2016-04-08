#!/usr/bin/env python
from collections import namedtuple
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import os,signal
from timer import Timer

import ipdb as pdb

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
  """ Sigmoid GLM mean for classification 
  Rescales sigmoid (probability) output [0,1]->[-1,1]
  """
  return tf.sub(tf.scalar_mul(2.0, tf.sigmoid(x)), tf.ones_like(x))

def logistic_loss_eltws(yp, y):
  """ Element-wise Logistic Loss. """
  return tf.log(tf.add(tf.exp(tf.neg(tf.mul(yp, y))), tf.ones_like(y)))

def square_loss_eltws(yp, y):
  """ Element-wise Square Loss. """
  return tf.scalar_mul(0.5, tf.square(tf.sub(yp, y)))

def multi_clf_err(yp, y):
  """ Multi-class classification Error """
  return tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(yp,1), tf.argmax(y,1)), tf.float32))

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

  def inference(self, children_preds, batch_size):
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
          ps = tf.tile(self.ps_b, tf.pack([batch_size, 1]))
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
  def __init__(self, dims, n_nodes, mean_types, loss_types, opt_types, eval_type=None):
    # dims: [input_d, out_d0, out_d1, ..., out_D==output_d]
    self.dims = dims
    self.n_nodes = n_nodes
    self.mean_types = mean_types
    self.loss_types = loss_types
    self.opt_types = opt_types
    self.eval_type = eval_type
    self.batch_size = tf.placeholder(tf.int32, shape=[])
    self.lr_boost = tf.placeholder(tf.float32, shape=[]) # placeholder for future changes
    self.lr_leaf = tf.placeholder(tf.float32, shape=[]) 

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
        l_preds = map(lambda i : l_nodes[i].inference(l_preds[i*nc:(i+1)*nc], self.batch_size), 
            range(n_nodes[i]))
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
        trainop_tgts = map(lambda nd : nd.training(self.lr_boost), l_nodes)
        l_train_ops, tgts = zip(*trainop_tgts)
        l_train_ops = list(itertools.chain.from_iterable(l_train_ops)) #flatten the list
        tgts = list(itertools.chain.from_iterable(tgts))
      else:
        l_train_ops = map(lambda x : x.training(self.lr_leaf), l_nodes) #every leaf node has one op.
        
      ll_train_ops.append(l_train_ops)
    #endfor
    #flatten all train_ops in one list
    self.train_ops = list(itertools.chain.from_iterable(ll_train_ops)) # dbg.training()
    # used to create checkpoints of the trained parameters, used for line search
    self.saver = tf.train.Saver()
    self.sigint_capture = False
    signal.signal(signal.SIGINT, self.signal_handler)
    print 'done.'
  #end __init__

  def fill_feed_dict(self, x, y, lr_boost, lr_leaf):
    if isinstance(x, list):
      b_size = len(x)
    else:
      b_size = x.shape[0]
    return { self.x_placeholder : x, self.y_placeholder : y,
             self.batch_size : b_size, 
             self.lr_boost : lr_boost, self.lr_leaf : lr_leaf}

  def inference(self):
    return self.pred

  def training(self):
    return self.train_ops

  def evaluation(self):
    if self.eval_type == None:
      return self.ll_nodes[-1][0].losses[-1]
    return self.eval_type(self.pred, self.y_placeholder)

  def signal_handler(self, signal, frame):
    print 'Interrupt Captured'
    self.sigint_capture = True

def main(_):
  # ------------- Dataset -------------
  arun_1d_regress = True 
  if arun_1d_regress:
    f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])
    data_set_size = 200000
    train_set = [pt for pt in dataset(data_set_size, f, 9122)]
    val_set = [pt for pt in dataset(201, f)]
    val_set = sorted(val_set, key = lambda x: x.x)
    x_val = [ pt.x for pt in val_set]
    y_val = [ pt.y for pt in val_set]
    model_name_suffix = '1d_reg'

    #n_nodes = [40, 20, 1]
    n_nodes = [20, 1]
    n_lvls = len(n_nodes)
    mean_types = [ sigmoid_clf_mean for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(square_loss_eltws)
    #opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = None

  else:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    train_set = list(range(mnist.train.num_examples))
    x_tra = mnist.train.images
    y_tra = mnist.train.labels
    x_val = mnist.validation.images # validation
    y_val = mnist.validation.labels
    model_name_suffix = 'mnist'

    n_nodes = [80, 40, 1]
    n_lvls = len(n_nodes)
    mean_types = [ sigmoid_clf_mean for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err

  input_dim = len(x_val[0].ravel())
  output_dim = len(y_val[0].ravel())

  dims = [output_dim for _ in xrange(n_lvls+1)] 
  dims[0] = input_dim
  dims[-1] = output_dim 

  # tuned for batch_size = 200, arun 1-d regress
  lr_boost_adam = 3e-4 #[50,1] #5e-3 [20,1]
  lr_leaf_adam = 3e-4 #8e-3

  #mnist lr
  lr_boost_adam = 3e-3
  lr_leaf_adam = 3e-3

  lr_boost = lr_boost_adam
  lr_leaf  = lr_leaf_adam
  gamma_boost = 0.7
  gamma_leaf = 0.7

  # modify the default tensorflow graph.
  dbg = TFDeepBoostGraph(dims, n_nodes, mean_types, loss_types, opt_types, eval_type)

  init = tf.initialize_all_variables()
  sess = tf.Session()
  print 'Initializing...'
  sess.run(init)
  print 'Initialization done'
  
  t = 0
  # As we can waste an epoch with the line search, max_epoch will be incremented when a line search
  # is done. However, to prevent infintie epochs, we set an ultimatum on the number of epochs
  # (max_epoch_ult) that stops this.
  epoch = -1
  max_epoch = np.Inf
  max_epoch_ult = max_epoch * 2 
  batch_size = 200
  val_interval = 500
  best_avg_loss = np.Inf 
  worsen_cnt = 0
  restore_threshold = len(train_set) / val_interval
  do_line_search = False
  min_non_ls_epochs = 4 # min. number of epochs in the beginning where we don't do line search
  model_dir = '../model/'
  if not os.path.isdir(model_dir):
      os.mkdir(model_dir)
  best_model_fname = 'best_model_{}.ckpt'.format(model_name_suffix) 
  init_model_fname = 'initial_model_{}.ckpt'.format(model_name_suffix) 
  best_model_path = os.path.join(model_dir, best_model_fname)
  init_model_path = os.path.join(model_dir, init_model_fname)
  dbg.saver.save(sess, init_model_path)
  stop_program = False
  while not stop_program and epoch < max_epoch and epoch < max_epoch_ult:
    epoch += 1
    print("-----Epoch {:d}-----".format(epoch))
    np.random.shuffle(train_set)
    for si in range(0, len(train_set), batch_size):
      #print 'train epoch={}, start={}'.format(epoch, si)
      si_end = min(si+batch_size, len(train_set))
      if arun_1d_regress:
        x = [ pt.x for pt in train_set[si:si_end] ]
        y = [ pt.y for pt in train_set[si:si_end] ]
      else:
        x = x_tra[train_set[si:si_end]]
        y = y_tra[train_set[si:si_end]]

      if dbg.sigint_capture == True:
         # don't do any work this iteration, restart all computation with the next
         break
      sess.run(dbg.training(), feed_dict=dbg.fill_feed_dict(x, y, lr_boost, lr_leaf))
      
      # Evaluate
      t += si_end-si
      if t % val_interval == 0:
        preds, avg_loss = sess.run([dbg.inference(), dbg.evaluation()], 
                                   feed_dict=dbg.fill_feed_dict(x_val, y_val, 
                                                                lr_boost, lr_leaf))
        assert(not np.isnan(avg_loss))
        # Plotting the fit.
        if arun_1d_regress:
          plt.figure(1)
          plt.clf()
          plt.plot(x_val, preds, label='Prediction')
          plt.plot(x_val, y_val, label='Ground Truth')
          plt.legend(loc=4)
          plt.draw()
          plt.show(block=False)
        print 'epoch={},t={} avg_loss={}'.format(epoch, t, avg_loss)

        if epoch < min_non_ls_epochs:
            continue
        
        if do_line_search:
            # restores if is worse than the best multiple times
            if avg_loss > best_avg_loss:
              worsen_cnt += 1
              if worsen_cnt > restore_threshold:
                print 'Restore to previous best loss: {}'.format(best_avg_loss)
                dbg.saver.restore(sess, best_model_path)
                worsen_cnt = 0
                max_epoch += 1
                lr_boost *= gamma_boost
                lr_leaf *= gamma_leaf
            else:
              worsen_cnt = 0
              lr_boost = lr_boost_adam
              lr_leaf = lr_leaf_adam
              dbg.saver.save(sess, best_model_path)
              best_avg_loss = avg_loss
    #endfor
    if dbg.sigint_capture == True:
      print("----------------------")
      print("Paused. Set parameters before loading the initial model again...")
      print("----------------------")
      pdb.set_trace()
      dbg.saver.restore(sess, init_model_path)
      epoch = -1 ; t = 0; 
      dbg.sigint_capture = False
  #endfor
  print("Program Finished")

if __name__ == '__main__':
  main(0)

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
def dataset_1d(num_pts, f, seed=None):
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

def tf_linear(name, l, dim, bias=False):
  with tf.name_scope(name):
    w1 = tf.Variable(
        tf.truncated_normal([dim[0], dim[1]],
                            stddev=3.0 / sqrt(float(dim[0]))),
        name='weights')
    l1 = tf.matmul(l, w1, name='linear')
    if not bias:
      return l1, [w1]
    b1 = tf.Variable(tf.truncated_normal([dim[1]], stddev=5.0 / sqrt(float(dim[0]))), name='biases')
    l2 = tf.nn.bias_add(l1, b1)
    return l2, [w1, b1]

def tf_linear_relu(name, l, dim, bias=False):
  with tf.name_scope(name):
    l1, v1 = tf_linear(name+'li', l, dim, bias)
    r1 = tf.nn.sigmoid(l1)
    return r1, v1

def tf_bottleneck(name, l, dim, last_relu=True):
  with tf.name_scope(name):
    r1, v1 = tf_linear_relu(name+'lr1', l, [dim[0], dim[1]], bias=True)
    l2, v2 = tf_linear(name+'l2', r1, [dim[1], dim[2]], bias=True)
    if last_relu:
      r2 = tf.nn.sigmoid(l2)
      return r2, v1+v2
    return l2, v1+v2

class TFLeafNode(object):
  """ Apply a K-dim generalized linear model 
      that maps the input x to K independently mapped dims.
  """

  def __init__(self, name, dim, mean_type, loss_type, opt_type, convert_y = True):
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
    self.convert_y = convert_y

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
      if self.convert_y:
        y_sgn = tf.sign(y, name='y_sgn') # target of dim
        y_abs = tf.abs(y, name='y_abs')  # wgt of each dim 

        # weighted average of loss_type(pred, y_sgn)
        ind_loss = self.loss_type(self.pred, y_sgn)
        wgt_loss = tf.mul(y_abs, ind_loss, 'wgt_loss')
        self.loss = tf.reduce_mean(wgt_loss, name='avg_loss')
      else:
        self.loss = tf.reduce_mean(self.loss_type(self.pred, y))
    return self.loss

  def training(self, y, lr):
    self.optimizer = self.opt_type(lr)
    compute_op = self.optimizer.compute_gradients(self.loss, [self.w, self.b])
    self.apply_op = [self.optimizer.apply_gradients(compute_op)]
    self.grads = [ op[0] for op in compute_op ] #gradient tensors in a list
    return self.grads, self.apply_op, []

class TFBottleneckLeafNode(object):
  """ Apply a bottleneck (resnet) that outputs K dimensions. 
      that maps the input x to K independently mapped dims.
  """

  def __init__(self, name, dim, reg_lambda, mean_type, loss_type, opt_type, convert_y=True):
    """ record params of the node for construction in the future.

    Args:
      dim: (D, K)
    """
    self.name = name
    self.dim = dim
    self.reg_lambda = reg_lambda
    self.mean_type = mean_type #e.g., tf.nn.relu
    self.loss_type = loss_type # element-wise loss 
    self.opt_type = opt_type   #e.g., tf.train.GradientDescentOptimizer
    self.convert_y = convert_y

  def inference(self, x):
    """ Construct inference part of the node: linear, relu, linear, relu 
      
    Args:
      x: input tensor, float - [batch_size, intermediate_dim, dim[0]]
    """
    bn, bn_var = tf_bottleneck(self.name + 'bn', x, self.dim, last_relu=False)
    bn_tf = self.mean_type(bn)
    li_tf, li_tf_var = tf_linear(self.name + 'li_tf', x, [self.dim[0], self.dim[-1]], bias = False)
    self.pred = bn_tf + li_tf
    self.variables = bn_var + li_tf_var
    return self.pred
      
  def loss(self, y):
    """ Construct TF loss graph given the inference graph.

    Args:
      y: target (sign and magnitude on each dim), float - [batch_size, dim[1]] 
    """
    with tf.name_scope(self.name):
      if self.convert_y:
        y_sgn = tf.sign(y, name='y_sgn') # target of dim
        y_abs = tf.abs(y, name='y_abs')  # wgt of each dim 
           
        # weighted average of loss_type(pred, y_sgn)
        ind_loss = self.loss_type(self.pred, y_sgn)
        wgt_loss = tf.mul(y_abs, ind_loss, 'wgt_loss')
        self.loss = tf.reduce_mean(wgt_loss, name='avg_loss')
      else:
        self.loss = tf.reduce_mean(self.loss_type(self.pred, y))

    #self.regularized_loss()
    return self.loss

  def regularized_loss(self):
    with tf.name_scope(self.name):
      regulation = 0.0
      for variable in self.variables:
        regulation += self.reg_lambda * tf.reduce_sum(tf.square(tf.reshape(variable,[-1])))
      self.regularized_loss = self.loss + regulation
    return self.regularized_loss

  def training(self, y, lr):
    self.optimizer = self.opt_type(lr)
    compute_op = self.optimizer.compute_gradients(self.loss, self.variables)
    self.apply_op = [self.optimizer.apply_gradients(compute_op)]
    self.grads = [ op[0] for op in compute_op ] #gradient tensors in a list
    return self.grads, self.apply_op, []

class TFBoostNode(object):
  def __init__(self, name, dim, reg_lambda, mean_type, loss_type, opt_type, ps_ws_val, batch_size, convert_y=True):
    """ record params of the node for construction in the future.

    Args:
      dim: (D, K)

      convert_y : if true, convert input target to -1, 1 with weights of 
        the original absolute value; else use input directly as learning targets. 
    """
    self.name = name
    self.dim = dim
    self.reg_lambda = reg_lambda
    self.mean_type = mean_type #e.g., tf.nn.relu
    self.loss_type = loss_type # element-wise loss 
    self.opt_type = opt_type   #e.g., tf.train.GradientDescentOptimizer
    self.convert_y = convert_y
    self.ps_ws_val = ps_ws_val
    self.batch_size = tf.to_float(batch_size)

  def inference(self, children_preds, batch_size):
    self.n_children = len(children_preds)
    with tf.name_scope(self.name):
      self.ps_b = tf.Variable(tf.zeros([1,self.dim[0]]), name='ps_bias')
      # learn ps_ws
      #self.ps_ws = [tf.Variable(tf.zeros([1]), name='ps_weight_'+str(ci)) 
      #    for ci in range(self.n_children)]
      # Set constant weight for all ps_ws
      # a list of ps_ws initialized using the given placeholder.
      #self.ps_ws = tf.fill([self.n_children], self.ps_ws_val, 'ps_weights')

      self.tf_w = tf.Variable(np.eye(self.dim[0], self.dim[1], dtype=np.float32), name='tf_weights')
      
      # TODO test version that doesn't store all partial sums
      self.psums = []
      self.y_hats = []
      for i in range(self.n_children+1):
        if i == 0:
          ps = tf.tile(tf.zeros([1,self.dim[0]]), tf.pack([batch_size, 1]))
        elif i==1:
          # The first weak learner directly predicts the target and thus is not weighted.
          ps = tf.add(ps, children_preds[i-1])
        else:
          # if we have a list of ps_val we can have a list of ps_ws #[i-1]))
          ps = tf.add(ps, tf.mul(children_preds[i-1], self.ps_ws_val / tf.to_float(i-1))) 
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
    #self.regularized_loss()
    return self.losses

  def regularized_loss(self):
    with tf.name_scope(self.name):
      self.regulation = self.reg_lambda *\
        (tf.reduce_sum(tf.square(self.tf_w),[0,1])+tf.reduce_sum(tf.square(self.ps_b)))
      self.regularized_losses = [ loss + self.regulation for loss in self.losses]
    return self.regularized_losses
      
  def training(self, y, lr):
    with tf.name_scope(self.name):
      # optimizer (n_children) is for bias
      compute_ops = []
      self.apply_ops = []
      self.children_tgts = []
      # Learning combination weights of weak learners using partial losses.
      for i in range(self.n_children+1):
        opt = self.opt_type(lr)        
        if i == 0:
          #compute_op = opt.compute_gradients(self.regularized_losses[i], var_list=[self.ps_b])
          #apply_op = opt.apply_gradients(compute_op)
          compute_op = None
          apply_op = None
        else:
          compute_op = None
          apply_op = None
          #if i == self.n_children:
          #  compute_op = opt.compute_gradients(self.regularized_losses[i], var_list=[self.tf_w])
          #  apply_op = opt.apply_gradients(compute_op)
          if i>1:
            grad_ps = tf.neg(tf.gradients(self.losses[i], [self.psums[i-1]])[0]) * self.batch_size
            sum_grads += grad_ps
            sum_y_hats += self.y_hats[i-1]
          else:
            grad_ps = y #- self.psums[i-1]
            sum_grads = y
            sum_y_hats = self.y_hats[0]
          tgt = sum_grads - sum_y_hats
          self.children_tgts.append(tgt)
        if compute_op is not None: # this implies apply op is not None
          compute_ops.append(compute_op)
          self.apply_ops.append(apply_op)
      #endfor 
      # list of (grads, varname)
      compute_ops = list(itertools.chain.from_iterable(compute_ops))
      self.grads = [ op[0] for op in compute_ops ]

      ## Compute the targets for the children
      #for i in range(1,self.n_children+1):
      #  grad_ps = tf.neg(tf.gradients(self.losses[i], [self.psums[i-1]])[0])
      #  self.children_tgts.append(grad_ps)
      ##endfor 
      ## compute_ops is list of (grads, varname)
      #opt = self.opt_type(lr) # construct the optimizer object 
      ## Learning the combining weights of weak learners (ps_ws using the final loss)
      ##compute_ops = opt.compute_gradients(self.losses[-1], var_list=[self.tf_w]+self.ps_ws) 
      ## Learning only the transformation weight using the final loss
      #compute_ops = opt.compute_gradients(self.losses[-1], var_list=[self.tf_w]) 
      ## apply_ops is an tensor flow operation to update variables 
      #self.apply_ops = [opt.apply_gradients(compute_ops)]
      #self.grads = [ op[0] for op in compute_ops ]
    return self.grads, self.apply_ops, self.children_tgts

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
    self.ps_ws_val = tf.placeholder(tf.float32, shape=[])
    self.reg_lambda = tf.placeholder(tf.float32, shape=[])

    # construct inference from bottom up
    print 'Construct inference()'
    self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]))
    self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]))
    self.ll_nodes = []
    dim_index = 0
    for i in range(len(n_nodes)):
      if i == 0:
        dim_index_end = dim_index+3
        dim = dims[dim_index:dim_index_end]; dim_index = dim_index_end-1

        l_nodes = [TFBottleneckLeafNode('leaf'+str(ni), dim, self.reg_lambda, 
            mean_types[i], loss_types[i], 
            opt_types[i], False) for ni in range(n_nodes[i])]  
        l_preds = map(lambda node : node.inference(self.x_placeholder), l_nodes)
      else:
        dim_index_end = dim_index+2
        dim = dims[dim_index:dim_index_end]; dim_index = dim_index_end-1

        #l_nodes = [TFBoostNode('boost'+str(ni), dim, mean_types[i], loss_types[i], 
        #    opt_types[i], self.ps_ws_val, i<len(n_nodes)-1) for ni in range(n_nodes[i])]  
        l_nodes = [TFBoostNode('boost'+str(ni), dim, self.reg_lambda, 
            mean_types[i], loss_types[i], 
            opt_types[i], self.ps_ws_val, self.batch_size, False) for ni in range(n_nodes[i])]  

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
    ll_compute_ops = []
    ll_apply_ops = []
    for i in reversed(range(len(n_nodes))):
      print 'depth={}'.format(i)
      l_nodes = self.ll_nodes[i]
      _ =  map(lambda ni : l_nodes[ni].loss(tgts[ni]), range(n_nodes[i])) 
      if i > 0:
        lr_lvl = self.lr_boost
      else:
        lr_lvl = self.lr_leaf
      #endif
      l_train_ops = zip(*map(lambda nd : nd.training(self.y_placeholder, lr_lvl), l_nodes))
      l_compute_ops, l_apply_ops, tgts = [ list(itertools.chain.from_iterable(ops)) 
          for ops in l_train_ops]
      ll_compute_ops.append(l_compute_ops)
      ll_apply_ops.append(l_apply_ops)
    #endfor
    #flatten all train_ops in one list
    self.train_ops = list(itertools.chain.from_iterable(ll_compute_ops + ll_apply_ops)) # dbg.training()
    #self.train_ops = list(itertools.chain.from_iterable(ll_apply_ops)) # dbg.training()
    # used to create checkpoints of the trained parameters, used for line search
    self.saver = tf.train.Saver()
    self.sigint_capture = False
    signal.signal(signal.SIGINT, self.signal_handler)
    print 'done.'
  #end __init__

  def fill_feed_dict(self, x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda):
    if isinstance(x, list):
      b_size = len(x)
    else:
      b_size = x.shape[0]
    feed_dict = { self.x_placeholder : x, self.y_placeholder : y,
                 self.batch_size : b_size, 
                 self.lr_boost : lr_boost, self.lr_leaf : lr_leaf,
                 self.ps_ws_val : ps_ws_val,
                 self.reg_lambda : reg_lambda}
    return feed_dict

  def inference(self):
    return self.pred

  def weak_learner_inference(self):
    return [nd.pred  for nd in self.ll_nodes[-2]]

  def training(self):
    return self.train_ops

  def evaluation(self, loss=False):
    if self.eval_type == None or loss:
      return self.ll_nodes[-1][0].losses[-1]
    return self.eval_type(self.pred, self.y_placeholder)

  def signal_handler(self, signal, frame):
    print 'Interrupt Captured'
    self.sigint_capture = True

def main(_):
  # ------------- Dataset -------------
  from textmenu import textmenu
  datasets = ['arun_1d', 'mnist', 'cifar']
  indx = textmenu(datasets)
  if indx == None:
      return
  dataset = datasets[indx]
  if dataset == 'arun_1d':
    f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])
    data_set_size = 200000
    all_data = [pt for pt in dataset_1d(data_set_size, f, 9122)]
    train_set = list(range(data_set_size))
    val_set = [pt for pt in dataset_1d(201, f)]
    val_set = sorted(val_set, key = lambda x: x.x)
    x_tra = np.expand_dims(np.hstack([ pt.x for pt in all_data]), 1)
    y_tra = np.expand_dims(np.hstack([ pt.y for pt in all_data]), 1)
    x_val = np.expand_dims(np.hstack([ pt.x for pt in val_set]), 1)
    y_val = np.expand_dims(np.hstack([ pt.y for pt in val_set]), 1)
    model_name_suffix = '1d_reg'

    #n_nodes = [40, 20, 1]
    n_nodes = [50, 1]
    n_lvls = len(n_nodes)
    mean_types = [ lambda x:x for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ square_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(square_loss_eltws)
    opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]
    #opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = None

    # tuned for batch_size = 200, arun 1-d regress
    lr_boost_adam = 1e-5 #[50,1] #5e-3 [20,1]
    lr_leaf_adam = 1e-7 #8e-3
    ps_ws_val = 1
    reg_lambda = 0.0

  elif dataset == 'mnist':
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    train_set = list(range(mnist.train.num_examples))
    x_tra = mnist.train.images
    y_tra = mnist.train.labels
    x_val = mnist.validation.images # validation
    y_val = mnist.validation.labels
    model_name_suffix = 'mnist'

    n_nodes = [80, 1]
    n_lvls = len(n_nodes)
    mean_types = [ sigmoid_clf_mean for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err
    
    #mnist lr
    lr_boost_adam = 3e-3
    lr_leaf_adam = 3e-3
    ps_ws_val = 1.0
    reg_lambda = 0.0

  elif dataset == 'cifar':
    data = np.load('/data/data/processed_cifar_resnet.npz')
    x_all = data['x_tra']; y_all = data['y_tra'];
    yp_all = data['yp_tra'];
 
    x_test = data['x_test']; y_test = data['y_test'];
    yp_test = data['yp_test'];

    # Adding the images themselves as features
    #x_all = np.hstack((x_all, data['im_train'][:,::5]))
    #x_test = np.hstack((x_test,data['im_test'][:,::5]))
    
    n_train = x_all.shape[0] 
    all_indices = np.arange(n_train)
    np.random.shuffle(all_indices)
    tra_val_split = 45000 #n_train * 9 // 10
    tra_indices = all_indices[:tra_val_split]
    val_indices = all_indices[tra_val_split:]

    x_tra = x_all[tra_indices]; y_tra = y_all[tra_indices] 
    x_val = x_all[val_indices]; y_val = y_all[val_indices]
    model_name_suffix = 'cifar10'
    
    n_nodes = [50, 1]
    n_lvls = len(n_nodes)
    mean_types = [ lambda x : x for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ square_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

    #opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err

    ##### Use all train and test:
    x_tra = x_all; y_tra = y_all; yp_tra = yp_all;
    x_val = x_test; y_val = y_test; yp_val = yp_test;

    train_set = list(range(x_tra.shape[0]))

    #cifar lr
    lr_boost_adam = 1e-3
    lr_leaf_adam = 1e-3
    ps_ws_val = 1.0
    reg_lambda = 1e-4

  input_dim = len(x_val[0].ravel())
  output_dim = len(y_val[0].ravel())

  dims = [output_dim for _ in xrange(n_lvls+2)] 
  dims[0] = input_dim
  dims[1] = input_dim # TODO do it in better style

  lr_boost = lr_boost_adam
  lr_leaf  = lr_leaf_adam


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
  batch_size = 10
  val_interval = 500

  # if line search, these will shrink learning rate until result improves. 
  do_line_search = False
  min_non_ls_epochs = 4 # min. number of epochs in the beginning where we don't do line search
  gamma_boost = 0.7
  gamma_leaf = 0.7
  # linesearch variables
  worsen_cnt = 0
  best_avg_loss = np.Inf 
  restore_threshold = len(train_set) / val_interval

  # Model saving paths. 
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
      x = x_tra[train_set[si:si_end]]
      y = y_tra[train_set[si:si_end]]
      
      if dbg.sigint_capture == True:
         # don't do any work this iteration, restart all computation with the next
         break
      sess.run(dbg.training(), 
            feed_dict=dbg.fill_feed_dict(x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda))
      
      # Evaluate
      t += si_end-si
      if si_end-si < batch_size: t = 0;
      if t % val_interval == 0:
        preds_tra, avg_loss_tra, avg_tgt_loss_tra =\
            sess.run([dbg.inference(), dbg.evaluation(), dbg.evaluation(loss=True)],
                feed_dict=dbg.fill_feed_dict(x_tra[:5000], y_tra[:5000], 
                                             lr_boost, lr_leaf, ps_ws_val, reg_lambda))
        preds, avg_loss, avg_tgt_loss = sess.run([dbg.inference(), dbg.evaluation(), dbg.evaluation(loss=True)], 
            feed_dict=dbg.fill_feed_dict(x_val, y_val, 
                                         lr_boost, lr_leaf, ps_ws_val, reg_lambda))
        assert(not np.isnan(avg_loss))
        # Plotting the fit.
        if dataset == 'arun_1d':
          weak_predictions = sess.run(dbg.weak_learner_inference(), 
            feed_dict=dbg.fill_feed_dict(x_val, y_val, 
                                         lr_boost, lr_leaf, ps_ws_val, reg_lambda))
          tgts = sess.run(dbg.ll_nodes[-1][0].children_tgts[2:], 
            feed_dict=dbg.fill_feed_dict(x_val, y_val, 
                                         lr_boost, lr_leaf, ps_ws_val, reg_lambda))
          plt.figure(1)
          plt.clf()
          plt.plot(x_val, y_val, label='Ground Truth')
          for wi, wpreds in enumerate(weak_predictions):
            plt.plot(x_val, wpreds, label=str(wi))
          #for wi, tgt in enumerate(tgts):
          #  plt.plot(x_val, tgt, label=str(wi))
          #plt.legend(loc=4)
          plt.plot(x_val, preds, lw=3, label='Prediction')
          plt.draw()
          plt.show(block=False)
        print 'epoch={},t={} \n avg_loss={} avg_tgt_loss={} \n loss_tra={} tgt_loss_tra={}'.format(epoch, t, avg_loss, avg_tgt_loss, avg_loss_tra, avg_tgt_loss_tra)

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
      # helper functions
      save_model = lambda fname : dbg.saver.save(sess, fname)
      save_best = partial(save_model, best_model_path)
      save_init = partial(save_model, init_model_path)
      pdb.set_trace()
      dbg.saver.restore(sess, init_model_path)
      epoch = -1 ; t = 0; 
      dbg.sigint_capture = False
  #endfor
  print("Program Finished")

if __name__ == '__main__':
  main(0)

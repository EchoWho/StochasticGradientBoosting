#!/usr/bin/env python
from collections import namedtuple
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import os,signal
from timer import Timer
import get_dataset 

import ipdb as pdb

import tensorflow as tf


from tf_deep_boost import *
from signal_handler import *


class MNISTAnytimeNNUtils(object):
  def __init__(self, params, dims, n_layers):
    self.params = params
    self.dims = dims
    self.n_layers = n_layers

  def preprocess(self, x):
    c1, vc1 = tf_conv_transform('conv_tf1', x, [5, 5, 1, 32], [1,1], tf.nn.relu, True) 

    c1 = tf.nn.max_pool(tf.reshape(c1, [-1,28,28,32]), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    c1 = tf.reshape(c1, [-1, 14*14*32])
    c2, vc2 = tf_conv_transform('conv_tf2', c1, [5,5,32,64], [1,1], tf.nn.relu, True)
    c2 = tf.nn.max_pool(tf.reshape(c2, [-1,14,14,64]), ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    c2 = tf.reshape(c2, [-1, 7*7*64])
    return c2

  def prediction(self, x):
    l_preds = []
    inter_dim = self.params['res_inter_dim']
    for i in range(self.n_layers):
      add_bias = (i == 0)
      fc1, fc1_var = tf_linear_transform('fc1_'+str(i), x, [self.dims[0], inter_dim], self.params['mean_type'], bias=add_bias) 
      fc2, fc2_var = tf_linear('fc2_'+str(i), fc1, [inter_dim, self.dims[1]], bias=add_bias)
      l_preds.append(fc2)
    return l_preds

class CIFARAnytimeNNUtils(object):
  def __init__(self, params, dims, n_layers):
    self.params = params
    self.dims = dims
    self.n_layers = n_layers

  def preprocess(self, x):
    pass

  def prediction(self, x):
    pass

class AnytimeNeuralNet(object):
  def __init__(self, n_layers, dims, utils_type, loss_type, opt_type, eval_type, utils_params):
    self.dims = dims
    self.loss_type = loss_type; self.opt_type = opt_type; self.eval_type = eval_type
    self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]), name='x_input')
    self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]), name='y_label')
    self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
    self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
    self.dropout_keep_prob = tf.placeholder(tf.float32, shape=(), name='dropout_keep_prob')
    self.dropout_weights = tf.ones([n_layers], name='ones_dropout_weight')
    self.dropout_flag = tf.nn.dropout(self.dropout_weights, self.dropout_keep_prob, name='dropout_weight')

    self.dataset_utils = utils_type(utils_params, dims, n_layers)
    self.x = self.dataset_utils.preprocess(self.x_placeholder)
    transformed_feat_dim = self.x.get_shape().as_list()[-1]
    dims[0] = transformed_feat_dim 

    self.l_preds = self.dataset_utils.prediction(self.x)

    # psums as predictions at each layer and loss for each layer and overall loss
    self.psums = []
    self.losses = []
    self.loss = 0
    for i in range(n_layers):
      if i == 0:
        self.psums.append(self.l_preds[0])
      else:
        self.psums.append(self.psums[-1] + self.l_preds[i])
      self.losses.append(tf.reduce_mean(loss_type(self.psums[-1], self.y_placeholder), name='loss'+str(i)))
      def do_compute_loss(z):
        if z <= 1 or z==n_layers-1:
          return True
        #primes = set([2,3,5,7, 11, 13, 17, 19, 23, 29, 
        #            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
        #            73, 79, 83, 89, 97,101,103,107,109,113, 
        #           127,131,137,139,149,151,157,163,167,173, 
        #            179,181,191,193,197,199,211,223,227,229, 
        #            233,239,241,251,257,263,269,271,277,281, 
        #            283,293,307,311,313,317,331,337,347,349, 
        #            353,359,367,373,379,383,389,397,401,409, 
        #            419,421,431,433,439,443,449,457,461,463, 
        #            467,479,487,491,499,503,509,521,523,541])
        exponential = set([0,1,3,7,15,31])
        reverse_exponential = set([31, 30, 28, 24, 16, 0])
        return z in reverse_exponential

      if do_compute_loss(i):
        print i
        self.loss += self.losses[i]
    self.pred = self.psums[-1]

    # optimization / training
    self.optimizer = self.opt_type(self.lr)
    self.train_op = self.optimizer.minimize(self.loss)

    self.eval_result = self.loss
    if self.eval_type is not None:
      self.eval_result = self.eval_type(self.pred, self.y_placeholder) 
    
    self.saver = tf.train.Saver()

  def inference(self):
    return self.pred

  def weak_learner_inference(self):
    return self.l_preds

  def optimization_loss(self):
    return self.loss
  
  def last_loss(self):
    return self.losses[-1]

  def training(self):
    return self.train_op

  def evaluation(self):
    return self.eval_result

  def fill_feed_dict(self, x, y, lr, kp=1.0):
    if isinstance(x, list):
      b_size = len(x)
    else:
      b_size = x.shape[0]
    feed_dict = { self.x_placeholder : x, self.y_placeholder : y,
                 self.batch_size : b_size, self.lr : lr, 
                 self.dropout_keep_prob : kp} 
    return feed_dict

def main():
  # ------------- Dataset -------------
  from textmenu import textmenu
  datasets = get_dataset.all_names()
  indx = textmenu(datasets)
  if indx == None:
      return
  dataset = datasets[indx]
  x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
  model_name_suffix = dataset

  # params
  if dataset == 'mnist':
    n_layers = 1 
    n_total_inter_dims = 1024
    lr = 1e-4
    dims = [x_tra.shape[1], y_tra.shape[1]]
    mean_type = tf.nn.relu
    loss_type = tf.nn.softmax_cross_entropy_with_logits
    opt_type = tf.train.AdamOptimizer
    eval_type = multi_clf_err 
    utils_params = {'res_inter_dim': n_total_inter_dims // n_layers, 'mean_type': mean_type}
    utils_type = MNISTAnytimeNNUtils
  elif dataset == 'cifar':
    n_layers = 5
    lr = 1e-4
    dims = [x_tra.shape[1], y_tra.shape[1]]
    mean_type = tf.nn.relu
    loss_type = tf.nn.softmax_cross_entropy_with_logits
    opt_type = tf.train.AdamOptimizer
    eval_type = multi_clf_err 
    utils_params = {'res_inter_dim': 128, 'mean_type': mean_type}
    utils_type = CIFARAnytimeNNUtils
    print "Not implemented"
    return -2

  ann = AnytimeNeuralNet(n_layers, dims, utils_type, loss_type, \
                         opt_type, eval_type, utils_params)

  # Model saving paths. 
  model_dir = '../model/'
  if not os.path.isdir(model_dir):
      os.mkdir(model_dir)
  best_model_fname = 'best_model_{}.ckpt'.format(model_name_suffix) 
  init_model_fname = 'initial_model_{}.ckpt'.format(model_name_suffix) 
  best_model_path = os.path.join(model_dir, best_model_fname)
  init_model_path = os.path.join(model_dir, init_model_fname)
  tf.train.SummaryWriter(logdir='../log/', graph=tf.get_default_graph())

  # sessions and initialization
  init = tf.initialize_all_variables()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
  print 'Initializing...'
  sess.run(init)
  print 'Initialization done'

  # Signal Handling:
  shandler = SignalHandler()

  # training epochs
  train_set = list(range(x_tra.shape[0]))
  batch_size = 50
  val_interval = batch_size * 40
  epoch = 0
  max_epoch = 100
  t=0
  while epoch < max_epoch:
    epoch += 1
    print("-----Epoch {:d}-----".format(epoch))
    np.random.shuffle(train_set)
    for si in range(0, len(train_set), batch_size):
      #print 'train epoch={}, start={}'.format(epoch, si)
      si_end = min(si+batch_size, len(train_set))
      x = x_tra[train_set[si:si_end]]
      y = y_tra[train_set[si:si_end]]

      if shandler.captured():
        break
      sess.run(ann.training(), feed_dict=ann.fill_feed_dict(x, y, lr, kp=0.5))
      
      # Evaluate
      t += si_end-si
      if si_end-si < batch_size: t = 0;
      if t % val_interval == 0:
        preds_tra, loss_tra, last_loss_tra, eval_tra = \
            sess.run([ann.inference(), ann.optimization_loss(), ann.last_loss(), ann.evaluation()],
                feed_dict=ann.fill_feed_dict(x_tra[:5000], y_tra[:5000],lr))
        ps_losses = sess.run(ann.losses, feed_dict=ann.fill_feed_dict(x_tra[:5000], y_tra[:5000],lr))
        plt.figure(1)
        plt.clf()
        plt.plot(np.arange(len(ps_losses)), np.log(ps_losses))
        plt.title('log scale loss vs. learner id')
        plt.draw()
        plt.show(block=False)
                                    
        assert(not np.isnan(loss_tra))
        preds_val, loss_val, last_loss_val, eval_val = \
            sess.run([ann.inference(), ann.optimization_loss(), ann.last_loss(), ann.evaluation()],
                feed_dict=ann.fill_feed_dict(x_val, y_val, lr))
        assert(not np.isnan(loss_val))
        
        # Plotting the fit.
        if dataset == 'arun_1d':
          weak_predictions = sess.run(ann.weak_learner_inference(), 
              feed_dict=ann.fill_feed_dict(x_val, y_val, lr)) 
          plt.figure(1)
          plt.clf()
          plt.plot(x_val, y_val, lw=3, color='green', label='GT')
          for wi, wpreds in enumerate(weak_predictions):
            plt.plot(x_val, wpreds, label='w'+str(wi))
          plt.plot(x_val, preds_val, lw=3, color='blue', label='Yhat')
          plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
          plt.title('AnytimeNeuralNet')
          plt.tight_layout()
          plt.draw()
          plt.show(block=False)
        print 'epoch={},t={} \n loss_val={} last_loss_val={} eval_val={}\n loss_tra={} last_loss_tra={} eval_tra={}'.format(epoch, t, loss_val, last_loss_val, eval_val, loss_tra, last_loss_tra, eval_tra)
    if shandler.captured():
      print("----------------------")
      print("Paused. Set parameters before loading the initial model again...")
      print("----------------------")
      # helper functions
      save_model = lambda fname : ann.saver.save(sess, fname)
      save_best = partial(save_model, best_model_path)
      save_init = partial(save_model, init_model_path)
      pdb.set_trace()
      ann.saver.restore(sess, init_model_path)
      epoch = -1 ; t = 0; 
      shandler.reset()

    #endfor
    # end of epoch, so save out the results so far

  pdb.set_trace()
  return 0

if __name__ == '__main__':
  main()

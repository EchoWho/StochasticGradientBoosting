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

class AnytimeNeuralNet(object):
  def __init__(self, n_layers, dims, mean_type, loss_type, opt_type, eval_type, weak_learner_params):
    self.dims = dims
    self.mean_type = mean_type; self.loss_type = loss_type; self.opt_type = opt_type;
    self.eval_type = eval_type
    self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]), name='x_input')
    self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]), name='y_label')
    self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
    self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')

    self.l_nodes = [TFLeafNode('leaf'+str(i), dims, 0, mean_type, None, None, False)
        for i in range(n_layers)]
    self.l_preds = map(lambda nd : nd.inference(self.x_placeholder, weak_learner_params), self.l_nodes)

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
      self.loss += self.losses[-1]
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

  def fill_feed_dict(self, x, y, lr):
    if isinstance(x, list):
      b_size = len(x)
    else:
      b_size = x.shape[0]
    feed_dict = { self.x_placeholder : x, self.y_placeholder : y,
                 self.batch_size : b_size, self.lr : lr } 
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
  if dataset == 'arun_1d':
    n_layers = 20
    lr = 1e-2
    dims = [x_tra.shape[1], y_tra.shape[1]]
    mean_type = sigmoid_clf_mean
    loss_type = square_loss_eltws
    opt_type = tf.train.AdamOptimizer
    eval_type = None
    weak_learner_params = {'type':'res', 'res_inter_dim':1, 'res_add_linear':False}
  elif dataset == 'mnist':
    total_conv = 250
    n_layers = 1 #50
    lr = 5e-3
    dims = [x_tra.shape[1], y_tra.shape[1]]
    mean_type = tf.nn.relu
    loss_type = tf.nn.softmax_cross_entropy_with_logits
    opt_type = tf.train.AdamOptimizer
    eval_type = multi_clf_err 
    weak_learner_params = {'type':'conv', 'filter_size':[5,5,1,total_conv//n_layers], 'stride':[2,2]}

  ann = AnytimeNeuralNet(n_layers, dims, mean_type, loss_type, \
                         opt_type, eval_type, weak_learner_params)

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
  batch_size = 64
  val_interval = batch_size * 10
  epoch = 0
  max_epoch = 200
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
      sess.run(ann.training(), feed_dict=ann.fill_feed_dict(x, y, lr))
      
      # Evaluate
      t += si_end-si
      if si_end-si < batch_size: t = 0;
      if t % val_interval == 0:
        preds_tra, loss_tra, last_loss_tra, eval_tra = \
            sess.run([ann.inference(), ann.optimization_loss(), ann.last_loss(), ann.evaluation()],
                feed_dict=ann.fill_feed_dict(x_tra[:5000], y_tra[:5000],lr))
                                             
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

  pdf.set_trace()
  return 0

if __name__ == '__main__':
  main()

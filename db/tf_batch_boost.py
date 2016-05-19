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

def main(_):
  # ------------- Dataset -------------
  from textmenu import textmenu
  datasets = get_dataset.all_names()
  indx = textmenu(datasets)
  if indx == None:
      return
  dataset = datasets[indx]
  x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
  model_name_suffix = dataset

  if dataset == 'arun_1d':
    n_nodes = [50, 1]
    n_lvls = len(n_nodes)
    mean_types = [tf.sin for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [square_loss_eltws for lvl in range(n_lvls-1) ]
    #loss_types = [logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(square_loss_eltws)
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = None

    weak_learner_params = {'type':'linear'}

    lr_boost_adam = 1e-3 
    lr_leaf_adam = 1e-2 
    lr_decay_step = x_tra.shape[0] * 3 
    ps_ws_val = 1.0
    reg_lambda = 0.0

  elif dataset == 'mnist':
    n_nodes = [10, 1]
    n_lvls = len(n_nodes)
    mean_types = [ tf.nn.relu for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err

    weak_learner_params = {'type':'conv', 'conv_size':[5,5], 'stride':[2,2]}
    
    #mnist lr
    lr_boost_adam = 1e-8
    lr_leaf_adam = 1e-3
    lr_decay_step = x_tra.shape[0] * 4 
    ps_ws_val = 1.0
    reg_lambda = 0.0

  elif dataset == 'cifar':
    n_nodes = [50, 1]
    n_lvls = len(n_nodes)
    mean_types = [ tf.sin for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ square_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

    #opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err

    ##### Use all train and test:
    x_tra = x_all; y_tra = y_all; yp_tra = yp_all;
    x_val = x_test; y_val = y_test; yp_val = yp_test;

    weak_learner_params = {'type':'linear'}

    #cifar lr
    lr_boost_adam = 1e-3
    lr_leaf_adam = 1e-2
    lr_decay_step = x_tra.shape[0] * 3 
    ps_ws_val = 0.5
    reg_lambda = 0 
  else:
    raise Exception('Did not recognize datset: {}'.format(dataset))

  train_set = list(range(x_tra.shape[0]))
  input_dim = len(x_val[0].ravel())
  output_dim = len(y_val[0].ravel())

  dims = [output_dim for _ in xrange(n_lvls+2)] 
  dims[0] = input_dim
  #dims[1] = input_dim # TODO do it in better style

  # modify the default tensorflow graph.
  weak_classification = True
  dbg = TFDeepBoostGraph(dims, n_nodes, weak_classification, mean_types, loss_types, opt_types,
          weak_learner_params, eval_type)

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
  max_epoch = 100
  max_epoch_ult = max_epoch * 2 
  batch_size = 64
  val_interval = 5000

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

  tf.train.SummaryWriter(logdir='../log/',graph=tf.get_default_graph())

  stop_program = False
  lr_gamma = 0.3 

  # Total number of samples
  global_step = 0
  tra_err = []
  val_err = []

  for learneri in range(1,n_nodes[0]+1):
    max_epoch = 12 
    epoch = -1
    t = 0 
    print("---------------------")
    print(" Weak learner: {:d}".format(learneri))
    # for a new weak learner, reset the learning rates
    lr_global_step = 0
    lr_boost = lr_boost_adam
    lr_leaf  = lr_leaf_adam
    while not stop_program and epoch < max_epoch:
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
        n_applies = len(dbg.training_update())
        
        if learneri == 0: # bias
          # ll_train_ops is a list of list of 3-tuples of (grads, apply_ops, child_tgts)
          # Each element of the 3-tuple is a list. 
          # 
          # Get the last node (boostnode a.k.a. root), and access its first gradient and first
          # apply ops, which are for the global bias. 
          # NVM
          #NVM ... when convert_y == weak_classification == False, ps_w and ps_b are not learned so this is
          # empty.
          train_op = [dbg.ll_train_ops[-1][0][0][0], dbg.ll_train_ops[-1][0][1][0]]
        else:
          # For each learneri = 1... ,n_nodes[0]+1,
          # we access the associated leaf node to get its gradients ans apply_ops
          train_op = dbg.ll_train_ops[0][learneri-1][0] + dbg.ll_train_ops[0][learneri-1][1]
        sess.run(train_op,
            feed_dict=dbg.fill_feed_dict(x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda))
        
        # Evaluate
        t += si_end-si
        if si_end-si < batch_size: t = 0;
        lr_global_step += si_end - si
        global_step += si_end - si
        if lr_global_step > lr_decay_step: 
          lr_global_step -= lr_decay_step 
          lr_boost *= lr_gamma
          lr_leaf *= lr_gamma
          print("----------------------")
          print('Decayed step size: lr_boost={:.3g}, lr_leaf={:.3g}'.format(lr_boost, lr_leaf))
          print("----------------------")
        if t % val_interval == 0:
          prediction_tensor = dbg.ll_nodes[-1][0].psums[learneri]
          tgt_loss_tensor = dbg.ll_nodes[-1][0].losses[learneri]
          preds_tra, avg_loss_tra, avg_tgt_loss_tra =\
              sess.run([prediction_tensor, dbg.evaluation(False, prediction_tensor), tgt_loss_tensor],
                  feed_dict=dbg.fill_feed_dict(x_tra[:5000], y_tra[:5000], 
                                               lr_boost, lr_leaf, ps_ws_val, reg_lambda))
          preds, avg_loss, avg_tgt_loss = \
              sess.run([prediction_tensor, dbg.evaluation(False, prediction_tensor), tgt_loss_tensor], 
                  feed_dict=dbg.fill_feed_dict(x_val, y_val, 
                                               lr_boost, lr_leaf, ps_ws_val, reg_lambda))

          tra_err.append( ( global_step, avg_loss_tra, avg_tgt_loss_tra) )
          val_err.append( ( global_step, avg_loss, avg_tgt_loss) )

          assert(not np.isnan(avg_loss))
          # Plotting the fit.
          if dataset == 'arun_1d':
            #weak_predictions = sess.run(dbg.weak_learner_inference(), 
            #  feed_dict=dbg.fill_feed_dict(x_val, y_val, 
            #                               lr_boost, lr_leaf, ps_ws_val, reg_lambda))
            #tgts = sess.run(dbg.ll_nodes[-1][0].children_tgts[2:], 
            #  feed_dict=dbg.fill_feed_dict(x_val, y_val, 
            #                               lr_boost, lr_leaf, ps_ws_val, reg_lambda))
            plt.figure(1)
            plt.clf()
            plt.plot(x_val, y_val, lw=3, color='green', label='Ground Truth')
            #for wi, wpreds in enumerate(weak_predictions):
            #  if wi==0:
            #    # recall the first one learns y directly.
            #    plt.plot(x_val, wpreds, label=str(wi))
            #  else:
            #    plt.plot(x_val, -wpreds, label=str(wi))
            #for wi, tgt in enumerate(tgts):
            #  plt.plot(x_val, tgt, label=str(wi))
            #plt.legend(loc=4)
            plt.plot(x_val, preds, lw=3, color='blue', label='Prediction')
            plt.draw()
            plt.show(block=False)
          print 'learner={},epoch={},t={} \n avg_loss={} avg_tgt_loss={} \n loss_tra={} tgt_loss_tra={}'.format(learneri, epoch, t, 
              avg_loss, avg_tgt_loss, avg_loss_tra, avg_tgt_loss_tra)

      #endfor
      save_fname = '../log/batch_err_vs_gstep_{:s}.npz'.format(model_name_suffix)
      np.savez(save_fname, tra_err=np.asarray(tra_err), val_err=np.asarray(val_err), learners=learneri) 
      print('Saved error rates to {}'.format(save_fname))
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

  save_fname = '../log/batch_err_vs_gstep_{:s}.npz'.format(model_name_suffix)
  np.savez(save_fname, tra_err=np.asarray(tra_err), val_err=np.asarray(val_err), learners=learneri) 
  print('Saved error rates to {}'.format(save_fname))
  pdb.set_trace()

if __name__ == '__main__':
  main(0)

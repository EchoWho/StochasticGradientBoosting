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

def main(_):
  # ------------- Dataset -------------
  from textmenu import textmenu
  datasets = get_dataset.all_names()
  indx = textmenu(datasets)
  if indx == None:
      return
  dataset = datasets[indx]
  if dataset == 'arun_1d':
    x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
    model_name_suffix = '1d_reg'

    #n_nodes = [40, 20, 1]
    n_nodes = [50, 1]
    n_lvls = len(n_nodes)
    mean_types = [tf.sin for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    #loss_types = [square_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types = [logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(square_loss_eltws)
    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = None

    weak_learner_params = {'type':'linear'}

    lr_boost_adam = 1e-3 
    lr_leaf_adam = 1e-2 
    ps_ws_val = 1.0
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

    n_nodes = [32, 1]
    n_lvls = len(n_nodes)
    mean_types = [ sigmoid_clf_mean for lvl in range(n_lvls-1) ]
    mean_types.append(lambda x : x)
    loss_types = [ logistic_loss_eltws for lvl in range(n_lvls-1) ]
    loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

    #lr = tf.train.exponential_decay( 
    #    learning_rate=1e-3, 
    #    global_step=tp.get_global_step_var(), 
    #    decay_steps=dataset_train.size() * 10, 
    #    decay_rate=0.3, staircase=True, name='learning_rate') 


    opt_types =  [ tf.train.AdamOptimizer for lvl in range(n_lvls) ]
    eval_type = multi_clf_err

    weak_learner_params = {'type':'conv', 'conv_size':[5,5], 'stride':[2,2]}
    
    #mnist lr
    lr_boost_adam = 1e-8
    lr_leaf_adam = 1e-3
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

    train_set = list(range(x_tra.shape[0]))

    weak_learner_params = {'type':'linear'}

    #cifar lr
    lr_boost_adam = 1e-3
    lr_leaf_adam = 1e-2
    ps_ws_val = 0.5
    reg_lambda = 0 

  input_dim = len(x_val[0].ravel())
  output_dim = len(y_val[0].ravel())

  dims = [output_dim for _ in xrange(n_lvls+2)] 
  dims[0] = input_dim
  #dims[1] = input_dim # TODO do it in better style

  lr_boost = lr_boost_adam
  lr_leaf  = lr_leaf_adam


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
  max_epoch = np.Inf
  max_epoch_ult = max_epoch * 2 
  batch_size = 64
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

  tf.train.SummaryWriter(logdir='../log/',graph=tf.get_default_graph())

  stop_program = False
  lr_gamma = 0.3 
  lr_decay_step = x_tra.shape[0] * 10.0
  global_step = 0

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
      n_applies = len(dbg.training_update())
      sess.run(dbg.training(),
          feed_dict=dbg.fill_feed_dict(x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda))
      
      # Evaluate
      t += si_end-si
      if si_end-si < batch_size: t = 0;
      global_step += si_end - si
      if global_step > lr_decay_step: 
        global_step -= lr_decay_step 
        lr_boost *= lr_gamma
        lr_leaf *= lr_gamma
        print("----------------------")
        print('Decayed step size: lr_boost={:.3g}, lr_leaf={:.3g}'.format(lr_boost, lr_leaf))
        print("----------------------")
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
          plt.plot(x_val, y_val, lw=3, color='green', label='Ground Truth')
          for wi, wpreds in enumerate(weak_predictions):
            if wi==0:
              # recall the first one learns y directly.
              plt.plot(x_val, wpreds, label=str(wi))
            else:
              plt.plot(x_val, -wpreds, label=str(wi))
          #for wi, tgt in enumerate(tgts):
          #  plt.plot(x_val, tgt, label=str(wi))
          #plt.legend(loc=4)
          plt.plot(x_val, preds, lw=3, color='blue', label='Prediction')
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

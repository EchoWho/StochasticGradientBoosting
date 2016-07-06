#!/usr/bin/env python
from collections import namedtuple
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import os,signal,sys
import numpy.linalg as la
from numpy.linalg import svd
from scipy.linalg import svd as scipy_svd
import numpy.random as random
import scipy.io
import ipdb as pdb
import tensorflow as tf


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'db'))
from textmenu import textmenu
from timer import Timer
from tf_deep_boost import *
import get_dataset

def random_feature_layer(x, filter_size, strides, rf_s):
  with tf.name_scope('random_feature'):
    rand_stat = random.get_state()
    
    W = tf.constant(random.normal(0, 1, filter_size) / rf_s / np.sqrt(2),
                    dtype=np.float32, name="rf_filter")
    B = tf.constant(random.uniform(0, 2*np.pi, filter_size[-1]), 
                    dtype=np.float32, shape=[1,1,filter_size[-1]], name="rf_filter_bias")

    convx = tf.nn.conv2d(x, W, strides=[1,strides[0],strides[1],1], 
                         padding="SAME", name="rf_conv")
    convx += B
    rand_feat = tf.sin(convx, name="rf_sin")
    return rand_feat


class OnlineMatrixSketch(object):
  # convert a matrix of (n, m) to (l, m)
  def __init__(self, l, m):
    self.feat_dim = m
    self.sketch_len = l
    self.sketch_mat = np.zeros([l, m],dtype=np.float32)
    self.empty_indx = 0
    self.mean = np.zeros(m, dtype=np.float32)

  def update(self, x_bat):
    n_samples = x_bat.shape[0]
    x_bat_indx = 0
    while self.empty_indx + n_samples > self.sketch_len:
      n_inserted = self.sketch_len - self.empty_indx 
      x_bat_indx_new = x_bat_indx + n_inserted
      self.sketch_mat[self.empty_indx:, :] = x_bat[x_bat_indx:x_bat_indx_new, :] - self.mean
      x_bat_indx = x_bat_indx_new

      try:
        U, s, Vh = svd(self.sketch_mat, full_matrices=False)
      except:
        U, s, Vh = scipy_svd(self.sketch_mat, full_matrices=False)

      s_len = s.shape[0]; half_ell = self.sketch_len // 2
      if s_len >= half_ell: 
        s[:half_ell] = np.sqrt(s[:half_ell]**2 - s[half_ell]**2)
        s[half_ell:] = 0.0
        self.sketch_mat[:half_ell, :] = np.dot(diag(s[:half_ell]), Vh[:half_ell,:])
        self.sketch_mat[half_ell:, :] = 0
        self.empty_indx = half_ell 

      
    n_inserted = n_samples - x_bat_indx
    if n_inserted > 0:
      empty_indx_new = self.empty_indx + n_inserted
      self.sketch_mat[self.empty_indx:empty_indx_new, :] = x_bat[x_bat_indx:, :] - self.mean
      self.empty_indx = empty_indx_new

class PCAConvolutionNet(object):

  def __init__(self, n_iters, x_dim, y_dim, l_filter_sizes, l_strides, l_rf_s, l_pca_dim):
    self.x_dim = x_dim
    self.y_dim = y_dim

    self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
    self.lr = tf.placeholder(tf.float32, shape=[], name='lr')
    self.x_placeholders = []
    self.random_features = []
    self.y_placeholder = tf.placeholder(tf.float32, shape=(None, y_dim), name='label')
      

    for i in range(n_iters+1): 
      x = tf.placeholder(tf.float32, 
              shape=(None, x_dim[0], x_dim[1], x_dim[2]), name='input'+str(i))
      self.x_placeholders.append(x)
      if i <= n_iters:
        rf_x = random_feature_layer(x, l_filter_sizes[i], l_strides[i], l_rf_s[i])
        self.random_features.append(rf_x)
      else:
        # prediction using the features
        dim = [ x_dim[0]*x_dim[1]*x_dim[2], y_dim ]
        x_flat = tf.reshape(x, [self.batch_size, -1])
        self.y_pred, self.pred_var = tf_linear_transform('linear_pred', 
            x_flat, dim, f=lambda x:x, bias=True) 
        self.loss = tf.mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, self.y_placeholder))
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.pred_var)
        
      # update dimension of the feature map (post pca)
      x_dim[0] /= l_strides[i][0]
      x_dim[1] /= l_strides[i][1]
      x_dim[2] = l_pca_dim[i]
    #end for
  #end __init__
      
  def inference(self, x_bat, y_bat, lr, sess, is_train=True):
    for i in range(n_iters):
      # compute random feature
      rf_x = self.random_features[i].eval(feed_dict=\
              {self.x_placeholders[i] : x_bat, self.batch_size : x_bat.shape[0]},
              session=sess)
      
      # compute PCA
      if is_train:
        is_mean_update = online_pca_update(rf_x)
      if is_mean_update:
        return None, None
      x_bat = online_pca_apply(rf_x)

    # prediction and update
    inference_operations = [self.pred, self.loss]
    if is_train:
      inference_operations.append(self.train_op)
    yp, loss = sess.run(inference_operations, feed_dict=\
        {self.lr:lr, self.y_placeholder : y_bat, 
         self.x_placeholders[i] : x_bat, self.batch_size : x_bat.shape[0]})
    return yp, loss
 
def main():
  #import sklearn.linear_model as lm

  datasets = get_dataset.all_names()
  indx = textmenu(datasets)
  if indx == None:
    return  
  dataset = datasets[indx]
  x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)


  d = np.load('/data/data/mnist.npz')
  X=d['X']
  Y=d['Y'].ravel()
  Xtest=d['Xtest']
  Ytest=d['Ytest'].ravel()
  print 'data loaded'

  filter_size = [5,5,1,200]
  stride = 2

  # sample patches to determine median of patch distance. 

  # sample filters (W, B) to create RBF

  # apply filters

  # sample patches to determine patch mean.

  # PCA patches. 




  #PCA first
  n_pca_dim = 50
  X_m = np.mean(X, axis=0) # mean
  X_zm = X - X_m # X with zero mean
  X_cov = X_zm.T.dot(X_zm) # X covariance
  eigval, eigvec = la.eig(X_cov)
  eigvec = eigvec[:, :n_pca_dim] # choose the dominanting 50 dimensions
  Xp = X.dot(eigvec)  # projections of X,Xtest to these 50 dim.
  Xtestp = Xtest.dot(eigvec)

  # Compute kernel step size s (median of dist among points)
  n_trials = int(Xp.shape[0]**1.5)
  I = random.randint(0, Xp.shape[0], n_trials)
  deltI = random.randint(1, Xp.shape[0], n_trials)
  J = (I + deltI) % X.shape[0]
  dists = sorted(map(lambda i : la.norm(Xp[I[i],:] - Xp[J[i],:]), range(n_trials)))
  s = dists[n_trials / 2]

  # generate rbf params
  n_rbf = 4000
  W = random.randn(Xp.shape[1], n_rbf) / s / np.sqrt(2)
  B = random.uniform(0, 2*np.pi, n_rbf)

  #Xf = np.cos(Xp.dot(W)+ B)
  #Xtestf = np.cos(Xtestp.dot(W)+B)

  np.savez('mnist_pca_rbf_param.npz', P=eigvec, W=W, B=B)
  np.savez('hw2_mnist.npz', X=X,Y=Y,Xtest=Xtest,Ytest=Ytest, P=eigvec, W=W,B=B)
  d2 = np.loadz('hw2_mnist.npz')
  scipy.io.savemat('hw2_mnist.mat', d2)

if __name__ == '__main__':
  main()

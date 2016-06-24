
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
import numpy.linalg as la
import numpy.random as random
import scipy.io

import ipdb as pdb

import tensorflow as tf


class PCAConvolutionNet(object):

  def __init__(self):
    pass


 
if __name__ == '__main__':
  #import sklearn.linear_model as lm
  d = np.load('/data/data/mnist.npz')
  X=d['X']
  Y=d['Y'].ravel()
  Xtest=d['Xtest']
  Ytest=d['Ytest'].ravel()

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

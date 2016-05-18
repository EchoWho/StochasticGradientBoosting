import numpy as np

import pdb

def all_names():
  return DATASETS.keys()

def get_dataset(name):
  if name not in DATASETS.keys():
    raise Exception('Unrecognized dataset {}. Choose from {}'.format(name, datasets.keys()))
  return DATASETS[name]()


def arun_1d(train_set_size = 200000, val_set_size = 200, random_seed = 9122):
  MAX_X=5
  f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])
  #rand_state = np.random.get_state() # in case we choose to save the old random state
  np.random.seed(random_seed)
  x_tra = -MAX_X+2.*MAX_X*np.random.rand(train_set_size)
  y_tra = f(x_tra)
  x_val = -MAX_X+2.*MAX_X*np.random.rand(val_set_size)
  x_val.sort()
  y_val = f(x_val)
  x_tra = np.expand_dims(x_tra, 1)
  x_val = np.expand_dims(x_val, 1)
  #np.random.set_state(rand_state) # reset the old random state
  return x_tra, y_tra.T, x_val, y_val.T

def mnist():
  pass

def cifar():
  pass

# DEFINE FUNCTIONS THAT RETURN EACH DATASET
DATASETS = {'arun_1d':arun_1d, 
            'mnist':mnist, 
            'cifar':cifar}

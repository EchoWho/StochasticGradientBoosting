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
  # original ARUN_1D
  sigmoid = lambda x : 1.0 / (1.0 + np.exp(-x))
  f = lambda x : np.array([8.*np.cos(x) + 2.5*x*np.sin(x) + 2.8*x])
  # single bump
  #f = lambda x: np.array([ 10*(sigmoid(3*(x+1.7)) + sigmoid(-3*(x-1.7))-1) ])
  # sudo sin wave
  #f = lambda x : np.array([ 10*(sigmoid(3*(x+3.4))+sigmoid(-3*(x+1.7))-sigmoid(3*(x-1.7))-sigmoid(-3*(x-3.4))) ])
  # square wave:
  #f = lambda x: np.array([ (x.astype(int) % 2 == 0).astype(np.float32)  ])
  #rand_state = np.random.get_state() # in case we choose to save the old random state
  np.random.seed(random_seed)
  x_tra = -MAX_X+2.*MAX_X*np.random.rand(train_set_size)
  y_tra = f(x_tra).T
  x_val = -MAX_X+2.*MAX_X*np.random.rand(val_set_size)
  x_val.sort()
  y_val = f(x_val).T
  x_tra = np.expand_dims(x_tra, 1)
  x_val = np.expand_dims(x_val, 1)
  #np.random.set_state(rand_state) # reset the old random state
  return x_tra, y_tra, x_val, y_val

def mnist():
  from tensorflow.examples.tutorials.mnist import input_data
  mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

  train_set = list(range(mnist.train.num_examples))
  x_tra = mnist.train.images
  y_tra = mnist.train.labels
  x_val = mnist.validation.images # validation
  y_val = mnist.validation.labels
  return x_tra, y_tra, x_val, y_val

def cifar(location = '/data/data/processed_cifar_resnet.npz'):
  data = np.load(location)
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
  return x_tra, y_tra, x_val, y_val

def grasp_hog(location='/data/data/GraspDataset/hog/grasp_hog.npz'):
  data = np.load(location)
  x_tra = data['x_train']; y_tra = data['y_train'];
  x_val = data['x_test']; y_val = data['y_test'];
  return x_tra, y_tra, x_val, y_val


def a9a_uci(location='/data/data/a9a/a9a_scaled_dataset.npz'):
  data = np.load(location)
  x_tra = data['x_tra']; y_tra = data['y_tra'];
  x_val = data['x_test']; y_val = data['y_test'];
  return x_tra, y_tra, x_val, y_val
 
def slice_uci(location='/data/data/slice/slice.npz'):
  data = np.load(location)
  x_all = data['x_tra']; y_all = data['y_tra'];
  n_train = x_all.shape[0] 
  all_indices = np.arange(n_train)
  np.random.shuffle(all_indices)
  tra_val_split = n_train * 9 // 10
  tra_indices = all_indices[:tra_val_split]
  val_indices = all_indices[tra_val_split:]
  x_tra = x_all[tra_indices]; y_tra = y_all[tra_indices] 
  x_val = x_all[val_indices]; y_val = y_all[val_indices]
  return x_tra, y_tra, x_val, y_val


# DEFINE FUNCTIONS THAT RETURN EACH DATASET
DATASETS = {'arun_1d':arun_1d, 
            'mnist':mnist, 
            'cifar':cifar,
            'grasp_hog':grasp_hog,
            'a9a':a9a_uci,
            'slice':slice_uci,
            }

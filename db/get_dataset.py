import numpy as np
#from tensorflow.models.image.cifar10 import cifar10_input
import get_cifar10_dataset as cifar10_input

import pdb


def label2onehot(labels, n_classes):
    n_samples = len(labels)
    onehot = np.zeros((n_samples, n_classes), dtype=np.float32)
    onehot[(np.arange(n_samples), labels)] = 1.0
    return onehot


class Dataset(object):

    def __init__(self):
        self.epoch = 0
        self.dims = [0, 0]

    def next_batch(self, batch_size):
        print "Not Implemented"
        pass

    def sample_training(self, size):
        print "Not Implemented"
        pass

    def next_validation(self, size):
        print "Not Implemented"
        pass

    def next_test(self, size):
        print "Not Implemented"
        pass


class CIFARDatasetTensorflow(Dataset):

    def __init__(self, batch_size=50, loc='/data/data/cifar/cifar10/cifar-10-batches-bin'):
        self.loc = loc
        img_size = cifar10_input.IMAGE_SIZE
        self.dims = [img_size * img_size * 3, 10]
        self.epoch = 0
        self.t = 0
        self.total_samples = 50000

        self.x_tra, self.y_tra = cifar10_input.distorted_inputs(data_dir=self.loc, batch_size=batch_size)
        self.x_val, self.y_val = cifar10_input.inputs(eval_data=False, data_dir=self.loc, batch_size=batch_size)
        self.x_tes, self.y_tes = cifar10_input.inputs(eval_data=True, data_dir=self.loc, batch_size=batch_size)

    def reshapes(self, x_y):
        x = x_y[0]
        y = x_y[1]
        return x.reshape([-1, self.dims[0]]), label2onehot(y, self.dims[1])

    def next_batch(self, batch_size, sess):
        self.t += batch_size
        if self.t >= self.total_samples:
            self.epoch += 1
            self.t %= self.total_samples
        return self.reshapes(sess.run([self.x_tra, self.y_tra]))

    def sample_training(self, batch_size, sess):
        return self.reshapes(sess.run([self.x_val, self.y_val]))

    def next_validation(self, batch_size):
        pass

    def next_test(self, batch_size, sess):
        return self.reshapes(sess.run([self.x_tes, self.y_tes]))


class CIFARDataset(Dataset):

    def __init__(self, loc='/data/data/cifar/cifar10/cifar-10-batches-py'):
        Dataset.__init__(self)
        self.loc = loc
        self.train_indx = 0
        self.train_fn_indx = 0
        self.data_dict = None
        self.load_new_dict = True
        self.train_fn_prefix = 'data_batch_'
        self.n_train_fn = 5
        self.test_fn = 'test_batch'
        self.data_dict = None
        self.test_data_dict = None
        self.dims = [3072, 10]

    def next_batch(self, size):
        if self.load_new_dict:
            self.load_new_dict = False
            self.train_indx = 0
            self.train_fn_indx = self.train_fn_indx % self.n_train_fn + 1
            self.data_dict = np.load(self.loc + '/' + self.train_fn_prefix + str(self.train_fn_indx))
            self.x_tra = self.data_dict['data'].reshape([-1, 3, 32, 32]).transpose([0, 2, 3, 1]).reshape([-1, 3072])
            self.y_tra = label2onehot(self.data_dict['labels'], 10)
            self.x_tra_means = np.mean(self.x_tra, axis=1)[:, np.newaxis]
            self.total_samples = self.data_dict['data'].shape[0]
            self.train_order = np.arange(self.total_samples, dtype=np.int32)
            # np.random.shuffle(self.train_order)

        old_indx = self.train_indx
        self.train_indx = min(self.train_indx + size, self.total_samples)
        self.load_new_dict = self.train_indx == self.total_samples
        if self.load_new_dict and self.train_fn_indx == self.n_train_fn:
            self.epoch += 1
        tra_indx = self.train_order[old_indx:self.train_indx]
        # return x_tra, y_tra
        return self.x_tra[tra_indx].astype(np.float32), self.y_tra[tra_indx]

    def sample_training(self, size):
        return self.x_tra[:size], self.y_tra[:size]

    def next_validation(self, size=None):
        return self.next_test(size)

    def next_test(self, size=None):
        if self.test_data_dict is None:
            self.test_data_dict = np.load(self.loc + '/' + self.test_fn)
            self.x_test = self.test_data_dict['data'].astype(np.float32).reshape(
                [-1, 3, 32, 32]).transpose([0, 2, 3, 1]).reshape([-1, 3072])
            self.x_test_mean = np.mean(self.x_test, axis=1)
            #self.x_test -= self.x_test_mean[:, np.newaxis].astype(np.float32)
            self.y_test = label2onehot(self.test_data_dict['labels'], 10)
            self.test_data_dict = 0
            self.test_indx = 0

        if size is None:
            size = self.x_test.shape[0]
        if self.test_indx == self.x_test.shape[0]:
            self.test_indx = 0
        old_indx = self.test_indx
        self.test_indx += size
        self.test_indx = min(self.test_indx, self.x_test.shape[0])
        return self.x_test[old_indx:self.test_indx], self.y_test[old_indx:self.test_indx]


class MNISTDataset(Dataset):

    def __init__(self):
        Dataset.__init__(self)
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

        self.x_tra = mnist.train.images
        self.y_tra = mnist.train.labels
        self.x_val = mnist.validation.images
        self.y_val = mnist.validation.labels
        self.x_tst = mnist.test.images
        self.y_tst = mnist.test.labels

        self.n_train = self.x_tra.shape[0]
        self.tra_indx = self.n_train
        self.train_order = np.arange(self.n_train)
        self.dims = [self.x_tra.shape[1], self.y_tra.shape[1]]

        self.val_indx = 0
        self.tst_indx = 0

    def next_batch(self, size):
        if self.tra_indx == self.n_train:
            np.random.shuffle(self.train_order)
            self.tra_indx = 0

        old_indx = self.tra_indx
        self.tra_indx = min(self.tra_indx + size, self.n_train)
        if self.tra_indx == self.n_train:
            self.epoch += 1
        batch_indx = self.train_order[old_indx:self.tra_indx]
        return self.x_tra[batch_indx], self.y_tra[batch_indx]

    def sample_training(self, size):
        return self.x_tra[:size], self.y_tra[:size]

    def next_validation(self, size=None):
        n_total = len(self.x_val)
        if size is None:
            size = n_total
        if self.val_indx == n_total:
            self.val_indx = 0
        old_indx = self.val_indx
        self.val_indx += size
        self.val_indx = min(n_total, self.val_indx)
        return self.x_val[old_indx:self.val_indx], self.y_val[old_indx:self.val_indx]

    def next_test(self, size=None):
        n_total = len(self.x_tst)
        if size is None:
            size = n_total
        if self.tst_indx == n_total:
            self.tst_indx = 0
        old_indx = self.tst_indx
        self.tst_indx += size
        self.tst_indx = min(n_total, self.tst_indx)
        return self.x_tst[old_indx:self.tst_indx], self.y_tst[old_indx:self.tst_indx]


def all_names():
    return DATASETS.keys()


def get_dataset(name):
    if name not in DATASETS.keys():
        raise Exception('Unrecognized dataset {}. Choose from {}'.format(name, datasets.keys()))
    return DATASETS[name]()


def arun_1d(train_set_size=200000, val_set_size=200, random_seed=9122):
    MAX_X = 5
    # original ARUN_1D
    sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))
    f = lambda x: np.array([8. * np.cos(x) + 2.5 * x * np.sin(x) + 2.8 * x])
    # single bump
    #f = lambda x: np.array([ 10*(sigmoid(3*(x+1.7)) + sigmoid(-3*(x-1.7))-1) ])
    # sudo sin wave
    #f = lambda x : np.array([ 10*(sigmoid(3*(x+3.4))+sigmoid(-3*(x+1.7))-sigmoid(3*(x-1.7))-sigmoid(-3*(x-3.4))) ])
    # square wave:
    #f = lambda x: np.array([ (x.astype(int) % 2 == 0).astype(np.float32)  ])
    # rand_state = np.random.get_state() # in case we choose to save the old random state
    np.random.seed(random_seed)
    x_tra = -MAX_X + 2. * MAX_X * np.random.rand(train_set_size)
    y_tra = f(x_tra).T
    x_val = sorted(-MAX_X + 2. * MAX_X * np.random.rand(val_set_size))
    y_val = f(x_val).T
    x_tra = np.expand_dims(x_tra, 1)
    x_val = np.expand_dims(x_val, 1)
    # np.random.set_state(rand_state) # reset the old random state
    return x_tra, y_tra, x_val, y_val


def mnist(validation=False):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('../data/MNIST_data', one_hot=True)

    train_set = list(range(mnist.train.num_examples))
    x_tra = mnist.train.images
    y_tra = mnist.train.labels
    if validation:
        x_val = mnist.validation.images  # validation
        y_val = mnist.validation.labels
    else:
        x_val = mnist.test.images
        y_val = mnist.test.labels
    return x_tra, y_tra, x_val, y_val


def cifar(location='/data/data/processed_cifar_resnet.npz', validation=False):
    data = np.load(location)
    x_all = data['x_tra']
    y_all = data['y_tra']
    yp_all = data['yp_tra']
    x_test = data['x_test']
    y_test = data['y_test']
    yp_test = data['yp_test']
    if not validation:
        return x_all, y_all, x_test, y_test
    # Adding the images themselves as features
    #x_all = np.hstack((x_all, data['im_train'][:,::5]))
    #x_test = np.hstack((x_test,data['im_test'][:,::5]))
    n_train = x_all.shape[0]
    all_indices = np.arange(n_train)
    np.random.shuffle(all_indices)
    tra_val_split = 45000  # n_train * 9 // 10
    tra_indices = all_indices[:tra_val_split]
    val_indices = all_indices[tra_val_split:]
    x_tra = x_all[tra_indices]
    y_tra = y_all[tra_indices]
    x_val = x_all[val_indices]
    y_val = y_all[val_indices]
    return x_tra, y_tra, x_val, y_val


def grasp_hog(location='/data/data/GraspDataset/hog/grasp_hog.npz'):
    data = np.load(location)
    x_tra = data['x_train']
    y_tra = data['y_train']
    x_val = data['x_test']
    y_val = data['y_test']
    return x_tra, y_tra, x_val, y_val


def a9a_uci(location='/data/data/a9a/a9a_scaled_dataset.npz'):
    data = np.load(location)
    x_tra = data['x_tra']
    y_tra = data['y_tra']
    x_val = data['x_test']
    y_val = data['y_test']
    return x_tra, y_tra, x_val, y_val


def slice_uci(location='/data/data/slice/slice.npz'):
    data = np.load(location)
    x_all = data['x_tra']
    y_all = data['y_tra']
    n_train = x_all.shape[0]
    all_indices = np.arange(n_train)
    np.random.shuffle(all_indices)
    tra_val_split = n_train * 9 // 10
    tra_indices = all_indices[:tra_val_split]
    val_indices = all_indices[tra_val_split:]
    x_tra = x_all[tra_indices]
    y_tra = y_all[tra_indices]
    x_val = x_all[val_indices]
    y_val = y_all[val_indices]
    return x_tra, y_tra, x_val, y_val


# DEFINE FUNCTIONS THAT RETURN EACH DATASET
DATASETS = {'arun_1d': arun_1d,
            'mnist': mnist,
            'cifar': cifar,
            'grasp_hog': grasp_hog,
            'a9a': a9a_uci,
            'slice': slice_uci,
            }

#!/usr/bin/env python
from collections import namedtuple
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import os
import sys
import signal
from timer import Timer
import get_dataset

import ipdb as pdb

import tensorflow as tf
import tensorflow.contrib.layers as layers


from tf_deep_boost import *
from signal_handler import *

def print_sameline(text):
    sys.stdout.write('{}\r'.format(text))
    sys.stdout.flush()


class MNISTAnytimeNNUtils(object):

    def __init__(self, params, dims, n_layers):
        self.params = params
        self.dims = dims
        self.n_layers = n_layers

    def preprocess(self, x):
        c1, vc1 = tf_conv_transform('conv_tf1', x, [5, 5, 1, 32], [1, 1], tf.nn.relu, True)

        c1 = tf.nn.max_pool(tf.reshape(c1, [-1, 28, 28, 32]), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        c1 = tf.reshape(c1, [-1, 14 * 14 * 32])
        c2, vc2 = tf_conv_transform('conv_tf2', c1, [5, 5, 32, 64], [1, 1], tf.nn.relu, True)
        c2 = tf.nn.max_pool(tf.reshape(c2, [-1, 14, 14, 64]), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        c2 = tf.reshape(c2, [-1, 7 * 7 * 64])
        return c2

    def prediction(self, x):
        l_preds = []
        inter_dim = self.params['res_inter_dim']
        for i in range(self.n_layers):
            add_bias = (i == 0)
            fc1, fc1_var = tf_linear_transform(
                'fc1_' + str(i), x, [self.dims[0], inter_dim], self.params['mean_type'], bias=add_bias)
            fc2, fc2_var = tf_linear('fc2_' + str(i), fc1, [inter_dim, self.dims[1]], bias=add_bias)
            l_preds.append(fc2)
        return l_preds


class ImageAnytimeNN2DUtils(object):

    def __init__(self, params, dims):
        self.params = {'pred_sum_gamma': 1.0}
        self.params.update(params)
        self.dims = dims

        self.channels = params['channels']  # [ 16, 32, 64 ]
        self.conv_kernel_sizes = params['conv_kernel']
        self.pool_kernel_sizes = params['pool_kernel']
        self.strides = params['strides']  # [ 1, 2, 2 ]
        self.depth = len(self.channels)
        self.width = params['width']
        self.mean_type = params['mean_type']
        self.init_channel = self.width * self.channels[0]

    def generate_one_feature(self, x, i, j, k):
        layer_type = self.params['layer_type'][i]
        if layer_type == 'conv':
            # assume x is an image [n, h,w,c]
            x = layers.convolution2d(x, num_outputs=self.channels[i],
                                     kernel_size=self.conv_kernel_sizes[i], stride=self.strides[i], padding='SAME',
                                     activation_fn=tf.identity, normalizer_fn=layers.batch_norm)
        elif layer_type == 'fc':
            # ensure feature is flattened.
            if len(x.get_shape().as_list()) > 2:
                x = tensor_to_feature(x)
            x = layers.fully_connected(x, num_outputs=self.channels[i],
                                       activation_fn=tf.identity, normalizer_fn=layers.batch_norm)
        return x

    def res_add(self, nx, px):
        new_dims = nx.get_shape().as_list()
        prev_dims = px.get_shape().as_list()
        if new_dims[-1] == prev_dims[-1]:
            return nx + px
        elif new_dims[-1] < prev_dims[-1]:
            # should not happen (we lost information by going to smaller channel size)
            print "expect channel size to grow and not shrink"
        else:
            # assert that image size shrinks by half, and the channel size doubles.
            assert(new_dims[1] * 2 == prev_dims[1] and new_dims[2] * 2 == prev_dims[2])
            assert(new_dims[3] == prev_dims[3] * 2)
            px = layers.max_pool2d(px, kernel_size=2, stride=2, padding='SAME')
            px = tf.concat(3, [px, tf.zeros_like(px, tf.float32)], name='res_zero_pad')
            return nx + px

    def feature_grid(self, x):
        ll_feats = []
        prev_l_feats = []
        for i in range(self.depth):
            l_feats = []
            for j in range(self.width):
                if i == 0:
                    feat = self.generate_one_feature(x, i, j, 0)
                else:
                    feat = 0
                    for k in range(j + 1):
                        feat += self.generate_one_feature(prev_l_feats[k], i, j, k)
                    if self.params['res_add'][i]:
                        feat = self.res_add(feat, ll_feats[i - 2][j])
                feat = self.mean_type(feat)
                l_feats.append(feat)
            ll_feats.append(l_feats)
            prev_l_feats = l_feats
        return ll_feats

    def weak_predictions(self, x):
        ll_feats = self.feature_grid(x)
        ll_preds = []
        for depth, l_feats in enumerate(ll_feats):
            l_preds = []
            for width, feat in enumerate(l_feats):
                if len(feat.get_shape().as_list()) > 2:
                    feat_map_size = feat.get_shape().as_list()[1]
                    feat = tf.nn.avg_pool(feat,
                        ksize=[1, feat_map_size, feat_map_size, 1], strides=[1,1,1,1], padding='VALID')
                    feat = tensor_to_feature(feat)
                f_dim = feat.get_shape().as_list()[-1]
                pred = tf_linear('fc_' + str(depth) + '_' + str(width), feat, [f_dim, self.dims[1]], bias=False)[0]
                if self.params['weak_predictions'] == 'individual':
                    l_preds.append(pred)
                elif self.params['weak_predictions'] == 'row_sum':
                    if width == 0:
                        psum = pred
                    else:
                        psum = psum * self.params['pred_sum_gamma'] + pred
                    l_preds.append(psum)
                elif self.params['weak_predictions'] == 'col_sum':
                    if depth == 0:
                        psum = pred
                    else:
                        psum = ll_preds[-1][width] * self.params['pred_sum_gamma'] + pred
                    l_preds.append(psum)
                elif self.params['weak_predictions'] == 'box_sum':
                    if depth == 0:
                        psum = 0
                    else:
                        psum = ll_preds[-1][width]
                    if width == 0:
                        row_psum = 0
                    row_psum += pred
                    psum += row_psum
                    l_preds.append(psum)
                elif self.params['weak_predictions'] == 'diag_sum':
                    if depth == 0 or width == len(l_feats) - 1:
                        psum = pred
                    else:
                        psum = ll_preds[-1][width + 1] + pred
                    l_preds.append(psum)
                # endif params check
            ll_preds.append(l_preds)
            # endif for width
        # endif for depth

        # Option 2: boosting: pred[i,j] = sum_{k <= j} f(feat[i,k])
        return ll_preds

    def anytime_predictions_and_losses(self, ll_preds, loss_type, y_gt):
        preds = [pred for l_preds in ll_preds for pred in l_preds]
        losses = [tf.reduce_mean(loss_type(pred, y_gt)) for pred in preds]
        return preds, losses


class AnytimeNeuralNet(object):

    def __init__(self, n_layers, dims, utils_type, loss_type, opt_type, eval_type, utils_params):
        self.dims = dims
        self.loss_type = loss_type
        self.opt_type = opt_type
        self.eval_type = eval_type
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
            self.losses.append(tf.reduce_mean(loss_type(self.psums[-1], self.y_placeholder), name='loss' + str(i)))
            # def do_compute_loss(z):
            #  if z <= 1 or z==n_layers-1:
            #    return True
            #  #primes = set([2,3,5,7, 11, 13, 17, 19, 23, 29,
            #  #            31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
            #  #            73, 79, 83, 89, 97,101,103,107,109,113,
            #  #           127,131,137,139,149,151,157,163,167,173,
            #  #            179,181,191,193,197,199,211,223,227,229,
            #  #            233,239,241,251,257,263,269,271,277,281,
            #  #            283,293,307,311,313,317,331,337,347,349,
            #  #            353,359,367,373,379,383,389,397,401,409,
            #  #            419,421,431,433,439,443,449,457,461,463,
            #  #            467,479,487,491,499,503,509,521,523,541])
            #  exponential = set([0,1,3,7,15,31])
            #  reverse_exponential = set([31, 30, 28, 24, 16, 0])
            #  return z in reverse_exponential

            # if do_compute_loss(i):
            #  print i
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
        feed_dict = {self.x_placeholder: x, self.y_placeholder: y,
                     self.batch_size: b_size, self.lr: lr}
        return feed_dict


class AnytimeNeuralNet2D(AnytimeNeuralNet):

    def __init__(self, dims, utils_type, loss_type, opt_type, eval_type, utils_params):
        self.dims = dims
        self.loss_type = loss_type
        self.opt_type = opt_type
        self.eval_type = eval_type
        self.batch_size = tf.placeholder(tf.int32, shape=[], name='batch_size')
        self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]), name='x_input')
        self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]), name='y_label')
        self.lr = tf.placeholder(tf.float32, shape=[], name='lr')

        self.dataset_utils = utils_type(utils_params, dims)
        img_channels = 3
        img_size = int(np.sqrt(dims[0] / img_channels))
        self.x = tf.reshape(self.x_placeholder, [-1, img_size, img_size, img_channels]) 
        self.ll_preds = self.dataset_utils.weak_predictions(self.x)
        self.anytime_preds, self.losses = self.dataset_utils.anytime_predictions_and_losses(
            self.ll_preds, loss_type, self.y_placeholder)
        self.pred = self.anytime_preds[-1]

        # Generate a loss for each prediction. 
        self.l_optimizers = [ self.opt_type(self.lr) for _ in self.anytime_preds ]
        self.l_train_ops = [ self.l_optimizers[i].minimize(loss) for i, loss in enumerate(self.losses) ]
        BASE_PROB = 0.3
        MAX_PROB = 1.0
        PROB_STEP = (MAX_PROB - BASE_PROB) / (len(self.losses) - 1)
        # Note that the last one is not included and it will always be picked by sample_train_ops
        self.l_train_op_probs = np.arange(BASE_PROB, MAX_PROB, PROB_STEP)
        self.l_train_op_probs /= np.sum(self.l_train_op_probs)
        self.l_train_op_probs = np.cumsum(self.l_train_op_probs)

        # Generate a single loss using sum of losses.
        self.loss = 0 
        total_weight = 0
        for lvl, prob in enumerate(self.l_train_op_probs): 
            if utils_params['res_add'][lvl// utils_params['width']]:
                self.loss += self.losses[lvl] * prob
                total_weight += prob
        self.loss += self.losses[-1] * total_weight

        # optimization / training
        self.optimizer = self.opt_type(self.lr)
        self.train_op = self.optimizer.minimize(self.loss)

        self.anytime_eval_results = self.losses
        if self.eval_type is not None:
            self.anytime_eval_results = map(lambda p: self.eval_type(p, self.y_placeholder), self.anytime_preds)
        self.eval_result = self.anytime_eval_results[-1]

        self.saver = tf.train.Saver()

    def training(self, method='random'):
        if method == 'static':
            return [ self.train_op ] 
        elif method == 'last':
            return [ self.l_train_ops[-1] ]
        elif method == 'random':
            p = np.random.uniform(0,1)
            indx = np.searchsorted(self.l_train_op_probs, p)
            return [ self.l_train_ops[indx], self.l_train_ops[-1] ]

def main():
    # ------------- Dataset -------------
    from textmenu import textmenu
    datasets = get_dataset.all_names()
    indx = textmenu(datasets)
    if indx is None:
        return
    dataset = datasets[indx]
    #x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
    model_name_suffix = dataset

    # default batch size
    batch_size = 50

    # params
    if dataset == 'mnist':
        n_layers = 1
        n_total_inter_dims = 1024
        lr = 1e-4
        dataset = get_dataset.MNISTDataset()
        dims = dataset.dims
        mean_type = tf.nn.relu
        loss_type = tf.nn.softmax_cross_entropy_with_logits
        opt_type = tf.train.AdamOptimizer
        eval_type = multi_clf_err
        #utils_params = {'res_inter_dim': n_total_inter_dims // n_layers, 'mean_type': mean_type}
        #utils_type = MNISTAnytimeNNUtils
        utils_params = {'image_side': 28, 'image_channels': 1, 'width': 4, 'channels': [8, 16],
                        'strides': [2, 2], 'mean_type': mean_type, 'weak_predictions': 'row_sum'}
        utils_type = ImageAnytimeNN2DUtils
    elif dataset == 'cifar':
        lr = 0.1
        batch_size = 128
        dataset = get_dataset.CIFARDatasetTensorflow(batch_size=batch_size)
        dims = dataset.dims
        mean_type = tf.nn.relu
        loss_type = tf.nn.softmax_cross_entropy_with_logits
        opt_type = lambda lr : tf.train.MomentumOptimizer(lr, momentum=0.9)
        eval_type = multi_clf_err

        def build_resnet_params(n=3, init_total_channel=32, width=2):
            channel = init_total_channel / width
            channels = [channel]
            layer_type = ['conv']
            res_add = [False]
            conv_kernel = [3]
            pool_kernel = [1]
            strides = [1]
            for i in range(3):  # feat map size shrink 4 times.
                for j in range(n * 2): # at each channel size, there are 2n conv
                    channels.append(channel)
                    layer_type.append('conv')
                    conv_kernel.append(3)
                    res_add.append(j % 2 == 1)
                    if channels[-1] != channels[-2]:
                        strides.append(2)
                        pool_kernel.append(2)
                    else:
                        strides.append(1)
                        pool_kernel.append(1)
                channel *= 2

            return {'width': width, 'channels': channels, 'layer_type': layer_type,
                    'res_add': res_add, 'conv_kernel': conv_kernel,
                    'pool_kernel': pool_kernel, 'strides': strides}

        utils_params = build_resnet_params()
        utils_params.update({'mean_type': mean_type, 'weak_predictions': 'row_sum', 
                             'pred_sum_gamma': 1.0})
        utils_type = ImageAnytimeNN2DUtils

    # ann = AnytimeNeuralNet(n_layers, dims, utils_type, loss_type, \
    #                       opt_type, eval_type, utils_params)
    ann = AnytimeNeuralNet2D(dims, utils_type, loss_type,
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
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    print 'Initializing...'
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)
    print 'Initialization done'

    # Signal Handling:
    shandler = SignalHandler()

    # training epochs
    val_interval = 48000
    max_epoch = 2000
    lr_decay_step = 80 # Never decays for adam? 
    lr_decay_gamma = 0.1
    t = 0
    last_epoch = -1

    # load saved_model HACK/TODO
    restore_model_path = None
    #restore_model_path = '../model/best_model_cifar.ckpt'
    #ann.saver.restore(sess, restore_model_path)
    while dataset.epoch < max_epoch:
        if last_epoch != dataset.epoch:
            print("-----Epoch {:d}-----\n".format(dataset.epoch))
            last_epoch = dataset.epoch
            if dataset.epoch > 0 and dataset.epoch % lr_decay_step == 0:
                lr *= lr_decay_gamma

        x, y = dataset.next_batch(batch_size, sess)
        actual_batch_size = x.shape[0]

        print_sameline('...epoch={},t={}'.format(dataset.epoch, t))
        sess.run(ann.training(method='last'), feed_dict=ann.fill_feed_dict(x, y, lr))

        # Evaluate
        t += actual_batch_size
        if t >= val_interval:
            t = 0
        if t % val_interval == 0 or (dataset.epoch == 0 and t == actual_batch_size and restore_model_path is not None):
            print '\n'
            #n_tra_eval_samples = 1000
            #n_tra_eval_batches = n_tra_eval_samples // batch_size
            ##x_tra_samples, y_tra_samples = dataset.sample_training(n_tra_eval_samples)
            #l_last_loss_tra = []
            #l_eval_tra = []
            #for tra_eval_i in range(n_tra_eval_batches):
            #    #indx_s = tra_eval_i * batch_size
            #    #indx_e = indx_s + batch_size
            #    x_tra_samples, y_tra_samples = dataset.sample_training(batch_size, sess)
            #    last_loss_tra, eval_tra = \
            #        sess.run([ann.last_loss(), ann.evaluation()],
            #                 feed_dict=ann.fill_feed_dict(x_tra_samples,  # [indx_s:indx_e],
            #                                              y_tra_samples, lr))  # [indx_s:indx_e], lr))
            #    assert(not np.isnan(last_loss_tra))
            #    l_last_loss_tra.append(last_loss_tra)
            #    l_eval_tra.append(eval_tra)
            #last_loss_tra = np.mean(l_last_loss_tra)
            #eval_tra = np.mean(l_eval_tra)

            #ps_losses = sess.run(ann.losses, feed_dict=ann.fill_feed_dict(x_tra_samples, y_tra_samples,lr))
            # plt.figure(1)
            # plt.clf()
            #plt.plot(np.arange(len(ps_losses)), np.log(ps_losses))
            #plt.title('log scale loss vs. learner id')
            # plt.draw()
            # plt.show(block=False)

            n_val = 5000
            n_val_batches = n_val // batch_size
            l_loss_val = []
            l_eval_val = []
            for vali in range(n_val_batches):
                x_val, y_val = dataset.next_test(batch_size, sess)
                run_ret = sess.run(ann.losses + ann.anytime_eval_results,
                             feed_dict=ann.fill_feed_dict(x_val, y_val, lr))
                n_ret = len(run_ret) // 2
                losses = run_ret[:n_ret] 
                evals = run_ret[n_ret:]

                assert(not np.isnan(losses[-1]))
                l_loss_val.append(losses)
                l_eval_val.append(evals)
            evals_val = np.mean(np.array(l_eval_val), axis=0)
            losses_val = np.mean(np.array(l_loss_val), axis=0)

            
            print 'epoch={},t={} \n evals_val:'.format(dataset.epoch, t)
            print evals_val
            print 'losses_val:'
            print losses_val
            #print 'last_loss_tra={} eval_tra={}\n'.format(last_loss_tra, eval_tra)
        # ENDIF evaluation
        if shandler.captured():
            print("----------------------")
            print("Paused. Set parameters before loading the initial model again...")
            print("----------------------")
            # helper functions
            save_model = lambda fname: ann.saver.save(sess, fname)
            save_best = partial(save_model, best_model_path)
            save_init = partial(save_model, init_model_path)
            pdb.set_trace()
            ann.saver.restore(sess, init_model_path)
            dataset.epoch = 0
            shandler.reset()
        # ENDIF handler of signals
    # end of epoch checks, so save out the results so far

    pdb.set_trace()
    return 0

if __name__ == '__main__':
    main()

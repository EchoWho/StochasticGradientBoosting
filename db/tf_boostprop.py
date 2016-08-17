#!/usr/bin/env python
from collections import namedtuple
import itertools
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, cos, sin
import os
import signal
from timer import Timer
import get_dataset

import ipdb as pdb

import tensorflow as tf

DataPt = namedtuple('DataPt', ['x', 'y'])
MAX_X = 5
# dataset generator function


def dataset_1d(num_pts, f, seed=None):
    if seed is not None:
        np.random.seed(seed)
    for _ in xrange(num_pts - 1):
        x = -MAX_X + 2. * MAX_X * np.random.rand(1)[0]
        y = f(x)
        yield DataPt(np.array([x]), y)
    yield DataPt(np.array([1]), f(1))

flags = tf.app.flags
FLAGS = flags.FLAGS


def sigmoid_clf_mean(x):
    """ Sigmoid GLM mean for classification
    Rescales sigmoid (probability) output [0,1]->[-1,1]
    """
    with tf.name_scope('sigmoid_clf_mean'):
        return tf.sub(tf.scalar_mul(2.0, tf.sigmoid(x)), tf.ones_like(x))


def logistic_loss_eltws(yp, y):
    """ Element-wise Logistic Loss. Assumes {-1, 1} targets"""
    return tf.log(tf.add(tf.exp(tf.neg(tf.mul(yp, y))), tf.ones_like(y)))


def logistic_loss_eltws_masked(yp, y):
    """ Element-wise Logistic Loss, masked by the value of y. Assumes {-1,1} targets with 0s meaning
      values to be masked """
    return tf.mul(tf.abs(y), tf.log(tf.add(tf.exp(tf.neg(tf.mul(yp, y))), tf.ones_like(y))))


def square_loss_eltws(yp, y):
    """ Element-wise Square Loss. """
    return tf.scalar_mul(0.5, tf.square(tf.sub(yp, y)))


def multi_clf_err(yp, y):
    """ Multi-class classification Error """
    return tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(yp, 1), tf.argmax(y, 1)), tf.float32))


def multi_clf_err_masked(yp, y):
    """ yp \in [-1, 1], y \in {-1,1} """
    yp_signed = 2.0 * tf.to_float(tf.greater_equal(yp, 0)) - 1.0
    eltws_correct = tf.mul(tf.abs(y), tf.to_float(tf.equal(yp_signed, y)))
    return tf.reduce_mean(tf.reduce_sum(eltws_correct, 1))


def tf_linear(name, l, dim, bias=False):
    feature_dim = int(l.get_shape()[-1])  # im_size * im_size
    with tf.name_scope(name):
        w1 = tf.Variable(
            tf.truncated_normal([feature_dim, dim[1]],
                                stddev=3.0 / sqrt(float(feature_dim))),
            name='weights')
        l1 = tf.matmul(l, w1, name='linear')
        if not bias:
            return l1, [w1]
        b1 = tf.Variable(tf.truncated_normal([dim[1]], stddev=5.0 / sqrt(float(feature_dim))), name='biases')
        l2 = tf.nn.bias_add(l1, b1)
        return l2, [w1, b1]


def tf_conv(name, l, conv_size, stride, bias=False):
    """
    Currently assumes 1 channel input SQUARE image (sqrt(len(x)) is assumed to be image size)
    :conv_size - [dim_x, dim_y] of the image (TODO: maybe support more channels)
    :stride - the stride in the image [stride_x, stride_y] for the convolution
    """
    feature_dim = int(l.get_shape()[-1])  # im_size * im_size
    im_size = int(sqrt(feature_dim))
    assert(im_size * im_size == feature_dim)
    input_channels = 1  # TODO: Maybe support RGB images
    output_channels = 5
    with tf.name_scope(name):
        l_image = tf.reshape(l, [-1, im_size, im_size, input_channels])
        w1 = tf.Variable(
            tf.truncated_normal([conv_size[0], conv_size[1], input_channels, output_channels],
                                stddev=3.0 / sqrt(float(conv_size[0]))),
            name='weights')
        l1 = tf.nn.conv2d(l_image, w1, strides=[1, stride[0], stride[1], 1], padding='SAME', name='conv')
        if not bias:
            return l1, [w1]
        b1 = tf.Variable(tf.truncated_normal([output_channels], stddev=5.0 /
                                             sqrt(float(output_channels))), name='biases')
        l2 = l1 + b1
        l2_vec = tf.reshape(l2, [-1, feature_dim / (stride[0] * stride[1]) * output_channels])
        return l2_vec, [w1, b1]


def tf_linear_transform(name, l, dim, f=lambda x: x, bias=False):
    with tf.name_scope(name):
        l1, v1 = tf_linear(name + 'li', l, dim, bias)
        r1 = f(l1)
        return r1, v1


def tf_conv_transform(name, l, dim, conv_size, stride, f=lambda x: x, bias=False):
    with tf.name_scope(name):
        c1, vc1 = tf_conv(name + 'ci', l, conv_size, stride, bias)
        #c2, vc2 = tf_conv(name+'c2i', tf.nn.relu(c1), np.array(conv_size)/2, stride, bias)
        l1, vl1 = tf_linear(name + 'li', f(c1), (None, dim[1]), bias)
        return l1, vl1 + vc1


def tf_bottleneck(name, l, dim, f=lambda x: x, last_transform=True):
    with tf.name_scope(name):
        r1, v1 = tf_linear_transform(name + 'lr1', l, [dim[0], dim[1]], f, bias=True)
        l2, v2 = tf_linear(name + 'l2', r1, [dim[1], dim[2]], bias=True)
        if last_transform:
            r2 = f(l2)
            return r2, v1 + v2
        return l2, v1 + v2


class TFLeafNode(object):
    """ Apply a K-dim generalized linear model
        that maps the input x to K independently mapped dims.
    """

    def __init__(self, name, dim, reg_lambda, mean_type, loss_type, opt_type, convert_y):
        """ record params of the node for construction in the future.

        Args:
          dim: (D, K)
        """
        self.name = name
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.mean_type = mean_type  # e.g., tf.nn.relu
        self.loss_type = loss_type  # element-wise loss
        self.opt_type = opt_type  # e.g., tf.train.GradientDescentOptimizer
        self.convert_y = convert_y

    def inference(self, x, weak_learner_params):
        """ Construct inference part of the node.

        Args:
          x: input tensor, float - [batch_size, dim[0]]
        """
        # linear transformation
        if weak_learner_params is None:
            learner_type = 'linear'
        else:
            learner_type = weak_learner_params['type']
        with tf.name_scope(self.name):
            if learner_type == 'linear':
                self.pred, self.variables = tf_linear_transform(self.name, x, self.dim, self.mean_type, True)
            elif learner_type == 'conv':
                conv_size = weak_learner_params['conv_size']
                stride = weak_learner_params['stride']
                self.pred, self.variables = tf_conv_transform(
                    self.name, x, self.dim, conv_size, stride, self.mean_type, True)
            else:
                raise Exception("Unrecognized weak_learner_params['type'] == {}".format(learner_type))
        return self.pred

    def loss(self, y):
        """ Construct TF loss graph given the inference graph.

        Args:
          y: target (sign and magnitude on each dim), float - [batch_size, dim[1]]
        """
        with tf.name_scope(self.name + '_loss'):
            if self.convert_y:
                y_sgn = tf.sign(y, name='y_sgn')  # target of dim
                y_abs = tf.abs(y, name='y_abs')  # wgt of each dim

                # weighted average of loss_type(pred, y_sgn)
                ind_loss = self.loss_type(self.pred, y_sgn)
                wgt_loss = tf.mul(y_abs, ind_loss, 'wgt_loss')
                self.loss = tf.reduce_mean(wgt_loss, name='avg_loss')
            else:
                self.loss = tf.reduce_mean(self.loss_type(self.pred, y))
        return self.loss

    def training(self, lr, loss=None):
        with tf.name_scope(self.name + '_training'):
            self.optimizer = self.opt_type(lr)
            if loss is None:
                loss = self.loss
            compute_op = self.optimizer.compute_gradients(loss, self.variables)
            self.apply_op = [self.optimizer.apply_gradients(compute_op)]
            self.grads = [op[0] for op in compute_op]  # gradient tensors in a list
            return self.grads, self.apply_op, []


class TFBottleneckLeafNode(object):
    """ Apply a bottleneck (resnet) that outputs K dimensions.
        that maps the input x to K independently mapped dims.
    """

    def __init__(self, name, dim, reg_lambda, mean_type, loss_type, opt_type, convert_y):
        """ record params of the node for construction in the future.

        Args:
          dim: (D, K)
        """
        self.name = name
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.mean_type = mean_type  # e.g., tf.nn.relu
        self.loss_type = loss_type  # element-wise loss
        self.opt_type = opt_type  # e.g., tf.train.GradientDescentOptimizer
        self.convert_y = convert_y

    def inference(self, x, weak_learner_params):
        """ Construct inference part of the node: linear, relu, linear, relu

        Args:
          x: input tensor, float - [batch_size, intermediate_dim, dim[0]]
        """

        if weak_learner_params is None:
            learner_type = 'linear'
        else:
            learner_type = weak_learner_params['type']

        if learner_type == 'linear':
            bn, bn_var = tf_bottleneck(self.name + 'bn', x, self.dim, self.mean_type, last_transform=False)
            #li_tf, li_tf_var = tf_linear(self.name + 'li_tf', x, [self.dim[0], self.dim[-1]], bias = False)
            self.pred = bn  # + li_tf
            self.variables = bn_var  # + li_tf_var
        elif learner_type == 'conv':
            conv_size = weak_learner_params['conv_size']
            stride = weak_learner_params['stride']
            self.pred, self.variables = tf_conv_transform(
                self.name + 'bn', x, self.dim, conv_size, stride, self.mean_type, True)
        else:
            raise Exception("Unrecognized weak_learner_params['type'] == {}".format(learner_type))

        return self.pred

    def loss(self, y):
        """ Construct TF loss graph given the inference graph.

        Args:
          y: target (sign and magnitude on each dim), float - [batch_size, dim[1]]
        """
        with tf.name_scope(self.name):
            if self.convert_y:
                y_sgn = tf.sign(y, name='y_sgn')  # target of dim
                y_abs = tf.abs(y, name='y_abs')  # wgt of each dim

                # weighted average of loss_type(pred, y_sgn)
                ind_loss = self.loss_type(self.pred, y_sgn)
                wgt_loss = tf.mul(y_abs, ind_loss, 'wgt_loss')
                self.loss = tf.reduce_mean(wgt_loss, name='avg_loss')
            else:
                self.loss = tf.reduce_mean(self.loss_type(self.pred, y))

        # self.regularized_loss()
        return self.loss

    def regularized_loss(self):
        with tf.name_scope(self.name + '_loss'):
            regulation = 0.0
            for variable in self.variables:
                regulation += self.reg_lambda * tf.reduce_sum(tf.square(tf.reshape(variable, [-1])))
            self.regularized_loss = self.loss + regulation
        return self.regularized_loss

    def training(self, lr, loss=None):
        with tf.name_scope(self.name + '_training'):
            self.optimizer = self.opt_type(lr)
            if loss is None:
                loss = self.loss
            compute_op = self.optimizer.compute_gradients(loss, self.variables)
            self.apply_op = [self.optimizer.apply_gradients(compute_op)]
            self.grads = [op[0] for op in compute_op]  # gradient tensors in a list
            return self.grads, self.apply_op, []


class TFBoostNode(object):

    def __init__(self, name, dim, reg_lambda, mean_type, loss_type,
                 opt_type, ps_ws_val, batch_size, convert_y, is_root=True):
        """ record params of the node for construction in the future.

        Args:
          dim: (D, K)

          convert_y : if true, convert input target to -1, 1 with weights of
            the original absolute value; else use input directly as learning targets.
        """
        self.name = name
        self.dim = dim
        self.reg_lambda = reg_lambda
        self.mean_type = mean_type  # e.g., tf.nn.relu
        self.loss_type = loss_type  # element-wise loss
        self.opt_type = opt_type  # e.g., tf.train.GradientDescentOptimizer
        self.convert_y = convert_y
        self.is_root = is_root
        self.ps_ws_val = ps_ws_val
        self.batch_size = tf.to_float(batch_size)

    def inference(self, children_preds, batch_size):
        self.n_children = len(children_preds)
        with tf.name_scope(self.name):
            self.ps_b = tf.Variable(tf.zeros([1, self.dim[0]]), name='ps_bias')
            # learn ps_ws if convert_y
            self.ps_ws = [tf.Variable(tf.ones([1]), name='ps_weight_' + str(ci))
                          for ci in range(self.n_children)]

            self.tf_w = tf.Variable(np.eye(self.dim[0], self.dim[1], dtype=np.float32), name='tf_weights')

            # TODO test version that doesn't store all partial sums
            self.psums = []
            self.y_hats = []
            for i in range(self.n_children + 1):
                if i == 0:
                    ps = tf.tile(self.ps_b, tf.pack([batch_size, 1]))
                # elif i==1:
                    # The first weak learner directly predicts the target and thus is not weighted.
                #  ps = ps + children_preds[i-1] * self.ps_ws[i-1]
                else:
                    ps = ps - children_preds[i - 1] * self.ps_ws[i - 1] * \
                        self.ps_ws_val  # / tf.to_float(sqrt(float(i+1)))
                self.psums.append(ps)
                self.y_hats.append(self.mean_type(tf.matmul(ps, self.tf_w)))
        return self.y_hats[-1]

    def loss(self, y):
        with tf.name_scope(self.name + '_loss'):
            if self.convert_y and not self.is_root:
                y_sgn = tf.sign(y, name='y_sgn')
                y_abs = tf.abs(y, name='y_abs')
                self.losses = [tf.reduce_mean(tf.mul(y_abs, self.loss_type(y_hat, y_sgn)))
                               for y_hat in self.y_hats]
            else:
                self.losses = [tf.reduce_mean(self.loss_type(y_hat, y)) for y_hat in self.y_hats]
        # self.regularized_loss()
        return self.losses

    def regularized_loss(self):
        with tf.name_scope(self.name):
            self.regulation = self.reg_lambda *\
                (tf.reduce_sum(tf.square(self.tf_w), [0, 1]) + tf.reduce_sum(tf.square(self.ps_b)))
            self.regularized_losses = [loss + self.regulation for loss in self.losses]
        return self.regularized_losses

    def training(self, lr, loss=None):
        with tf.name_scope(self.name + '_training'):
            # optimizer (n_children) is for bias
            compute_ops = []
            self.apply_ops = []
            self.children_tgts = []
            # Learning combination weights of weak learners using partial losses.
            for i in range(self.n_children + 1):
                opt = self.opt_type(lr)

                # Children weights
                if self.convert_y:
                    if i == 0:
                        compute_op = opt.compute_gradients(self.losses[i], var_list=[self.ps_b])
                    else:
                        compute_op = opt.compute_gradients(self.losses[i], var_list=[self.ps_ws[i - 1]])
                    compute_ops.append(compute_op)
                    apply_op = opt.apply_gradients(compute_op)
                    self.apply_ops.append(apply_op)

                # Transformation matrix tf_w BTW THIS WRONG regarding convert_y TODO
                # if i == self.n_children:
                #  compute_op = opt.compute_gradients(self.regularized_losses[i], var_list=[self.tf_w, self.ps_ws[i-1]])
                #  apply_op = opt.apply_gradients(compute_op)

                if i > 0:
                    tgt = self.losses[-1]  # i] # TODO
                    self.children_tgts.append(tgt)
            # endfor
            # list of (grads, varname)
            compute_ops = list(itertools.chain.from_iterable(compute_ops))
            self.grads = [op[0] for op in compute_ops]

        return self.grads, self.apply_ops, self.children_tgts


class TFDeepBoostGraph(object):

    def __init__(self, dims, n_nodes, weak_classification, mean_types,
                 loss_types, opt_types, weak_learner_params=None, eval_type=None):
        """
        : weak_classification - bool - If True the weak learners are weighted classifications to
            fit their targets (parent gradients w.r.t. previous partial sum). This implies the
            parent need to learn the weak learner weights.
            If False, the weak learners directly regress for the targets, and parents do not
            learn the weights and use a constant combination weight (ps_ws_val as placeholder).
        """
        # dims: [input_d, out_d0, out_d1, ..., out_D==output_d]
        self.dims = dims
        self.n_nodes = n_nodes
        self.weak_classification = weak_classification
        self.mean_types = mean_types
        self.loss_types = loss_types
        self.opt_types = opt_types
        self.eval_type = eval_type
        self.batch_size = tf.placeholder(tf.int32, shape=[])
        self.lr_boost = tf.placeholder(tf.float32, shape=[])  # placeholder for future changes
        self.lr_leaf = tf.placeholder(tf.float32, shape=[])
        self.ps_ws_val = tf.placeholder(tf.float32, shape=[])
        self.reg_lambda = tf.placeholder(tf.float32, shape=[])

        # construct inference from bottom up
        print 'Construct inference()'
        self.x_placeholder = tf.placeholder(tf.float32, shape=(None, dims[0]))
        self.y_placeholder = tf.placeholder(tf.float32, shape=(None, dims[-1]))
        self.ll_nodes = []
        dim_index = 0
        for i in range(len(n_nodes)):
            if i == 0:
                dim_index_end = dim_index + 3
                dim = dims[dim_index:dim_index_end]
                dim_index = dim_index_end - 1

                l_nodes = [TFBottleneckLeafNode('leaf' + str(ni), dim, self.reg_lambda,
                                                mean_types[i], loss_types[i],
                                                opt_types[i], self.weak_classification) for ni in range(n_nodes[i])]
                l_preds = map(lambda node: node.inference(self.x_placeholder, weak_learner_params), l_nodes)
            else:
                dim_index_end = dim_index + 2
                dim = dims[dim_index:dim_index_end]
                dim_index = dim_index_end - 1

                # l_nodes = [TFBoostNode('boost'+str(ni), dim, mean_types[i], loss_types[i],
                #    opt_types[i], self.ps_ws_val, i<len(n_nodes)-1) for ni in range(n_nodes[i])]
                is_root = i + 1 == len(n_nodes)
                l_nodes = [TFBoostNode('boost' + str(ni), dim, self.reg_lambda,
                                       mean_types[i], loss_types[i],
                                       opt_types[i], self.ps_ws_val, self.batch_size, self.weak_classification, is_root) for ni in range(n_nodes[i])]

                assert(n_nodes[i - 1] % n_nodes[i] == 0)
                nc = n_nodes[i - 1] / n_nodes[i]  # n_children
                l_preds = map(lambda i: l_nodes[i].inference(l_preds[i * nc:(i + 1) * nc], self.batch_size),
                              range(n_nodes[i]))
            # endif
            self.ll_nodes.append(l_nodes)
        # endfor
        self.pred = l_preds[0]  # dbg.inference()

        print 'Construct loss() and training()'
        # construct loss and training_op from top down
        tgts = [self.y_placeholder]  # prediction target of nodes on a level (back to front)
        ll_compute_ops = []
        ll_apply_ops = []
        self.ll_train_ops = []
        for i in reversed(range(len(n_nodes))):
            print 'depth={}'.format(i)
            l_nodes = self.ll_nodes[i]
            if i > 0:
                lr_lvl = self.lr_boost
                map(lambda ni: l_nodes[ni].loss(tgts[ni]), range(n_nodes[i]))
                l_train_ops = map(lambda nd: nd.training(lr_lvl), l_nodes)
            else:
                lr_lvl = self.lr_leaf
                l_train_ops = map(lambda ni: l_nodes[ni].training(lr_lvl, tgts[ni]), range(n_nodes[i]))
            # endif
            self.ll_train_ops.append(l_train_ops)
            l_compute_ops, l_apply_ops, tgts = [list(itertools.chain.from_iterable(ops))
                                                for ops in zip(*l_train_ops)]
            ll_compute_ops.append(l_compute_ops)
            ll_apply_ops.append(l_apply_ops)
        # endfor
        # flatten all train_ops in one list
        self.ll_train_ops = list(reversed(self.ll_train_ops))
        self.compute_ops = list(itertools.chain.from_iterable(ll_compute_ops))  # dbg.training()
        self.apply_ops = list(itertools.chain.from_iterable(ll_apply_ops))  # dbg.training()
        self.train_ops = list(itertools.chain.from_iterable(ll_compute_ops + ll_apply_ops))  # dbg.training()
        # used to create checkpoints of the trained parameters, used for line search
        self.sigint_capture = False
        signal.signal(signal.SIGINT, self.signal_handler)
        self.saver = tf.train.Saver()
        print 'done.'
    # end __init__

    def fill_feed_dict(self, x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda):
        if isinstance(x, list):
            b_size = len(x)
        else:
            b_size = x.shape[0]
        feed_dict = {self.x_placeholder: x, self.y_placeholder: y,
                     self.batch_size: b_size,
                     self.lr_boost: lr_boost, self.lr_leaf: lr_leaf,
                     self.ps_ws_val: ps_ws_val,
                     self.reg_lambda: reg_lambda}
        return feed_dict

    def inference(self):
        return self.pred

    def weak_learner_inference(self):
        return [nd.pred for nd in self.ll_nodes[-2]]

    def training(self):
        return self.train_ops

    def training_compututation(self):
        return self.compute_ops

    def training_update(self):
        return self.apply_ops

    def evaluation(self, loss=False, prediction=None):
        if self.eval_type is None or loss:
            return self.ll_nodes[-1][0].losses[-1]
        if prediction is None:
            prediction = self.pred
        return self.eval_type(prediction, self.y_placeholder)

    def signal_handler(self, signal, frame):
        print 'Interrupt Captured'
        self.sigint_capture = True


def main(_):
    from textmenu import textmenu
    # ------------- Dataset -------------
    datasets = get_dataset.all_names()
    indx = textmenu(datasets)
    if indx is None:
        return
    dataset = datasets[indx]
    x_tra, y_tra, x_val, y_val = get_dataset.get_dataset(dataset)
    model_name_suffix = dataset
    if dataset == 'arun_1d':
        n_nodes = [200, 1]
        n_lvls = len(n_nodes)
        mean_types = [sigmoid_clf_mean for lvl in range(n_lvls - 1)]
        mean_types.append(lambda x: x)
        loss_types = [square_loss_eltws for lvl in range(n_lvls - 1)]
        #loss_types = [logistic_loss_eltws for lvl in range(n_lvls-1) ]
        loss_types.append(square_loss_eltws)
        opt_types = [tf.train.AdamOptimizer for lvl in range(n_lvls)]
        eval_type = None

        weak_learner_params = {'type': 'linear'}

        lr_boost_adam = 0.3 * 1e-2
        lr_leaf_adam = 0.3 * 1e-1
        lr_decay_step = x_tra.shape[0] * 10
        ps_ws_val = 1.0
        reg_lambda = 0.0

    elif dataset == 'mnist':
        n_nodes = [10, 1]
        n_lvls = len(n_nodes)
        mean_types = [tf.nn.relu for lvl in range(n_lvls - 1)]
        mean_types.append(lambda x: x)
        loss_types = [logistic_loss_eltws for lvl in range(n_lvls - 1)]
        loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

        opt_types = [tf.train.AdamOptimizer for lvl in range(n_lvls)]
        eval_type = multi_clf_err

        weak_learner_params = {'type': 'conv', 'conv_size': [5, 5], 'stride': [2, 2]}

        # mnist lr
        lr_boost_adam = 1e-8
        lr_leaf_adam = 1e-3
        lr_decay_step = x_tra.shape[0] * 5
        ps_ws_val = 1.0
        reg_lambda = 0.0

    elif dataset == 'grasp_hog':
        n_nodes = [32, 1]
        n_lvls = len(n_nodes)
        mean_types = [sigmoid_clf_mean for lvl in range(n_lvls - 1)]
        mean_types.append(lambda x: x)
        loss_types = [logistic_loss_eltws for lvl in range(n_lvls - 1)]
        loss_types.append(logistic_loss_eltws_masked)

        opt_types = [tf.train.AdamOptimizer for lvl in range(n_lvls)]
        eval_type = multi_clf_err_masked
        weak_learner_params = {'type': 'linear'}

        lr_boost_adam = 1e-3
        lr_leaf_adam = 1e-2
        lr_decay_step = x_tra.shape[0] * 3
        ps_ws_val = 1.0
        reg_lambda = 0.0

    elif dataset == 'cifar':
        n_nodes = [50, 1]
        n_lvls = len(n_nodes)
        mean_types = [tf.sin for lvl in range(n_lvls - 1)]
        mean_types.append(lambda x: x)
        loss_types = [square_loss_eltws for lvl in range(n_lvls - 1)]
        loss_types.append(tf.nn.softmax_cross_entropy_with_logits)

        #opt_types =  [ tf.train.GradientDescentOptimizer for lvl in range(n_lvls) ]
        opt_types = [tf.train.AdamOptimizer for lvl in range(n_lvls)]
        eval_type = multi_clf_err

        # Use all train and test:
        x_tra = x_all
        y_tra = y_all
        yp_tra = yp_all
        x_val = x_test
        y_val = y_test
        yp_val = yp_test

        train_set = list(range(x_tra.shape[0]))

        weak_learner_params = {'type': 'linear'}

        # cifar lr
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

    dims = [output_dim for _ in xrange(n_lvls + 2)]
    dims[0] = input_dim
    dims[1] = input_dim  # TODO do it in better style

    lr_boost = lr_boost_adam
    lr_leaf = lr_leaf_adam

    # modify the default tensorflow graph.
    weak_classification = False
    dbg = TFDeepBoostGraph(dims, n_nodes, weak_classification, mean_types, loss_types, opt_types,
                           weak_learner_params, eval_type)

    init = tf.initialize_all_variables()
    #sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
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
    val_interval = batch_size * 10

    # if line search, these will shrink learning rate until result improves.
    do_line_search = False
    min_non_ls_epochs = 4  # min. number of epochs in the beginning where we don't do line search
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

    tf.train.SummaryWriter(logdir='../log/', graph=tf.get_default_graph())

    stop_program = False
    lr_gamma = 0.3
    lr_global_step = 0

    # Total number of samples
    global_step = 0
    tra_err = []
    val_err = []

    while not stop_program and epoch < max_epoch and epoch < max_epoch_ult:
        epoch += 1
        print("-----Epoch {:d}-----".format(epoch))
        np.random.shuffle(train_set)
        for si in range(0, len(train_set), batch_size):
            # print 'train epoch={}, start={}'.format(epoch, si)
            si_end = min(si + batch_size, len(train_set))
            x = x_tra[train_set[si:si_end]]
            y = y_tra[train_set[si:si_end]]

            if dbg.sigint_capture == True:
                # don't do any work this iteration, restart all computation with the next
                break
            n_applies = len(dbg.training_update())
            sess.run(dbg.training(),
                     feed_dict=dbg.fill_feed_dict(x, y, lr_boost, lr_leaf, ps_ws_val, reg_lambda))

            # Evaluate
            t += si_end - si
            if si_end - si < batch_size:
                t = 0
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
                preds_tra, avg_loss_tra, avg_tgt_loss_tra =\
                    sess.run([dbg.inference(), dbg.evaluation(), dbg.evaluation(loss=True)],
                             feed_dict=dbg.fill_feed_dict(x_tra[:5000], y_tra[:5000],
                                                          lr_boost, lr_leaf, ps_ws_val, reg_lambda))
                assert(not np.isnan(avg_loss_tra))
                preds, avg_loss, avg_tgt_loss = sess.run([dbg.inference(), dbg.evaluation(), dbg.evaluation(loss=True)],
                                                         feed_dict=dbg.fill_feed_dict(x_val, y_val,
                                                                                      lr_boost, lr_leaf, ps_ws_val, reg_lambda))
                assert(not np.isnan(avg_loss))

                tra_err.append((global_step, avg_loss_tra, avg_tgt_loss_tra))
                val_err.append((global_step, avg_loss, avg_tgt_loss))

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
                    plt.plot(x_val, y_val, lw=3, color='green', label='GT')
                    for wi, wpreds in enumerate(weak_predictions):
                        plt.plot(x_val, -wpreds, label=str(wi))
                    # for wi, tgt in enumerate(tgts):
                    #  plt.plot(x_val, tgt, label=str(wi))
                    plt.plot(x_val, preds, lw=3, color='blue', label='Yhat')
                    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                    plt.title('boostprop')
                    plt.tight_layout()
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
        # endfor
        # end of epoch, so save out the results so far
        np.savez('../log/err_vs_gstep_fr_{:s}.npz'.format(model_name_suffix),
                 tra_err=np.asarray(tra_err), val_err=np.asarray(val_err))
        if dbg.sigint_capture == True:
            print("----------------------")
            print("Paused. Set parameters before loading the initial model again...")
            print("----------------------")
            # helper functions
            save_model = lambda fname: dbg.saver.save(sess, fname)
            save_best = partial(save_model, best_model_path)
            save_init = partial(save_model, init_model_path)
            pdb.set_trace()
            dbg.saver.restore(sess, init_model_path)
            epoch = -1
            t = 0
            dbg.sigint_capture = False
    # endfor

    print("Program Finished")
    np.savez('../log/err_vs_gstep_{:s}.npz'.format(model_name_suffix),
             tra_err=np.asarray(tra_err), val_err=np.asarray(val_err))
    pdb.set_trace()

if __name__ == '__main__':
    main(0)

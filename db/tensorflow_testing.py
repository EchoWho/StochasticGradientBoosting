#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor, sqrt, pi
import tensorflow as tf

from IPython import embed


def weight_variable(shape):
    #initial = tf.truncated_normal(shape, stddev=30.5)
    initial = tf.random_normal(shape, stddev=0.1)
    #initial = tf.zeros(shape)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    #initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

X_RANGE = (-5, 5)


def f(x):
    return 8. * np.cos(x) + 2.5 * x * np.sin(x) + 2.8 * x


def data_gen(num_pts):
    x = X_RANGE[0] + (X_RANGE[1] - X_RANGE[0]) * np.random.rand(num_pts)
    y = f(x)
    return np.reshape(x, (-1, 1)), np.reshape(y, (-1, 1))

if __name__ == "__main__":
    nodes_per_layer = [20]
    num_layers = len(nodes_per_layer)
    x_dim = 1  # feature dim
    y_dim = 1  # target dim
    l2_lambda = 0
    num_batches = 2500  # number of passes through training data

    num_data_pts = 150
    num_test = 200
    # tf.set_random_seed(1)
    # np.random.seed(1)
    test_x, test_y = data_gen(num_test)
    sort_order = np.argsort(test_x.T).ravel()
    test_x = test_x[sort_order]
    test_y = test_y[sort_order]

    # None indicates that the first dimension, corresponding to the batch size, can be of any size
    x = tf.placeholder(tf.float32, shape=[None, x_dim])
    y_ = tf.placeholder(tf.float32, shape=[None, y_dim])
    Ws = []
    Bs = []
    Ls = []
    x_prev = x
    print('Setting up layers')
    for layer_num in range(num_layers + 1):
        input_dim = x_prev.get_shape()[1].value
        if layer_num < num_layers:
            output_dim = nodes_per_layer[layer_num]
        else:
            output_dim = y_dim
        W = weight_variable([input_dim, output_dim])
        B = bias_variable([output_dim])

        linear = tf.nn.xw_plus_b(x_prev, W, B)
        if layer_num < num_layers:
            #linear = tf.nn.l2_normalize(linear, 1)
            #layer = tf.nn.relu(linear)
            layer = tf.nn.sigmoid(linear)
            #layer = tf.nn.tanh(linear)
        else:
            layer = linear
        Ws.append(W)
        Bs.append(B)
        Ls.append(layer)
        x_prev = layer
    # endfor
    print('Setting up regularization')
    # compute regularization
    for i, (W, B) in enumerate(zip(Ws, Bs)):
        if i == 0:
            regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(B)
        else:
            regularization += tf.nn.l2_loss(W) + tf.nn.l2_loss(B)
    # endfor
    y = Ls[-1]  # output of last layer is our regression outptut
    # indiv_loss = tf.nn.l2_loss(y_-y) # computes sum() over all points
    indiv_loss = tf.scalar_mul(0.5, tf.square(tf.sub(y_, y)))

    mean_loss = tf.reduce_mean(indiv_loss)
    total_loss = mean_loss + l2_lambda * regularization
    global_step = tf.Variable(0, name='global_step', trainable=False)  # Counter
    #optimizer = tf.train.GradientDescentOptimizer(1e-2)
    #optimizer = tf.train.FtrlOptimizer(1e0)
    #optimizer = tf.train.MomentumOptimizer(1e-5, 0.2)
    #optimizer = tf.train.AdagradOptimizer(step_size)
    #optimizer = tf.train.RMSPropOptimizer(5e-3)
    #optimizer = tf.train.AdamOptimizer(3e-2, 0.9, 0.9999999, 1e-6)
    optimizer = tf.train.AdamOptimizer(8e-2)
    train_step = optimizer.minimize(total_loss, global_step=global_step)

    init_op = tf.initialize_all_variables()

    print('Starting training...')
    # start the training
    sess = tf.Session()
    sess.run(init_op)
    train_loss = np.zeros(num_batches)
    for i in range(num_batches):
        train_x, train_y = data_gen(num_data_pts)
        train_dict = {x: train_x, y_: train_y}
        _, losses = sess.run([train_step, total_loss], feed_dict=train_dict)
        if i % 50 == 0:
            test_dict = {x: test_x, y_: test_y}
            preds, test_loss = sess.run([y, total_loss], feed_dict=test_dict)
            print 'my tensorflow dnn: {:.3g}'.format(test_loss)
        train_loss[i] = losses

    test_dict = {x: test_x, y_: test_y}
    preds, test_loss = sess.run([y, total_loss], feed_dict=test_dict)
    print 'my tensorflow dnn: {:.3g}'.format(test_loss)

    sess.close()

    #`print('Starting skflow...')
    #`import skflow
    #`dnn = skflow.TensorFlowDNNRegressor(hidden_units=nodes_per_layer)
    #`#dnn = skflow.TensorFlowDNNRegressor(hidden_units=nodes_per_layer,
    #`#        batch_size=num_data_pts, steps=50, max_to_keep=0)
    #`for i in range(5):
    #`    train_x, train_y = data_gen(num_data_pts)
    #`    dnn.partial_fit(train_x, train_y)
    #`preds_skflow = dnn.predict(test_x)
    #err = preds_skflow-test_y; l2_err = 0.5*err*err
    # print 'skFlow dnn: {}'.format(l2_err.mean())

    plt.plot(test_x, test_y, label='Ground Truth')
    #plt.plot(test_x, preds_skflow, label='skflow')
    plt.plot(test_x, preds, label='my tensorflow')
    plt.legend(fontsize=14, loc=2)
    plt.show()

    #print('Starting linear regression...')
    #from sklearn.linear_model.ridge import Ridge
    #r = Ridge(l2_lambda)
    #r.fit(train_x, train_y)
    #pred_y = r.predict(test_x)
    #err = pred_y-test_y; l2_err = 0.5*err*err
    # print 'linear reg batch: {}'.format(l2_err.mean())

    #print('Starting sklearn mlp regression...')
    # embed()
    #from sklearn.neural_network import MLPRegressor
    #skmlp = MLPRegressor(nodes_per_layer, activation='relu', alpha=l2_lambda),
    # for i in range(num_batches):
    #    train_x, train_y = data_gen(num_data_pts)
    #    skmlp.partial_fit(train_x, train_y)
    #pred_y = dnn.predict(test_x)
    #err = pred_y-test_y; l2_err = 0.5*err*err
    # print 'skmlp: {}'.format(l2_err.mean())

    # embed()

    print('Done...')

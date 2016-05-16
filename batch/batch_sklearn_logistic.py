#!/usr/bin/env python
import numpy as np
import ipdb as pdb
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


#discrete sin wave
X = np.arange(100)
X = np.array(X, dtype=np.float32)
X = X[:, np.newaxis]
X = np.hstack([X, np.zeros_like(X)])
Y = X[:,0] % 2


lr = LogisticRegression()
lr.fit(X, Y)
Ypred_sklr = lr.predict_proba(X)

plt.plot(X[:, 0], Ypred_sklr, lw=2, label='Sklearn.LogisticRegression')


import tensorflow as tf
def tf_linear(name, l, dim, bias=False):
  with tf.name_scope(name):
    w1 = tf.Variable(
        tf.truncated_normal([dim[0], dim[1]],
                            stddev=3.0 / np.sqrt(float(dim[0]))),
        name='weights')
    l1 = tf.matmul(l, w1, name='linear')
    if not bias:
      return l1, [w1]
    b1 = tf.Variable(tf.truncated_normal([dim[1]], stddev=5.0 / np.sqrt(float(dim[0]))), name='biases')
    l2 = tf.nn.bias_add(l1, b1)
    return l2, [w1, b1]

def tf_linear_transform(name, l, dim, f=lambda x:x, bias=False):
  with tf.name_scope(name):
    l1, v1 = tf_linear(name+'li', l, dim, bias)
    r1 = f(l1)
    return r1, v1


def tf_bottleneck(name, l, dim, f=lambda x:x, last_transform=True):
  with tf.name_scope(name):
    r1, v1 = tf_linear_transform(name+'lr1', l, [dim[0], dim[1]], f, bias=True)
    l2, v2 = tf_linear(name+'l2', r1, [dim[1], dim[2]], bias=True)
    if last_transform:
      r2 = f(l2)
      return r2, v1+v2
    return l2, v1+v2


X_pl = tf.placeholder(tf.float32, shape=[None,2])
Y_pl = tf.placeholder(tf.float32, shape=[None,1])

n_btlnck = 2
for i in range(n_btlnck):
  bn, bn_var = tf_bottleneck('btlnck0', X_pl, [2,1,1], tf.nn.sigmoid, last_transform=False)
  if i == 0:
    pred = bn
  else:
    pred += bn

loss = tf.reduce_mean(tf.squared_difference(Y_pl,pred))
optimizer = tf.train.AdamOptimizer(1e-1)
training = optimizer.minimize(loss)


init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)

indices = np.arange(X.shape[0])
for iter in range(200):
  np.random.shuffle(indices)
  for i in range(X.shape[0]):
    sess.run(training, feed_dict={X_pl : X[indices[i,np.newaxis], :], Y_pl : Y[indices[i],np.newaxis,np.newaxis]})

  if iter % 30 == 29:
    Ypred_tf = sess.run(pred, feed_dict={X_pl : X, Y_pl : Y[:, np.newaxis]})
    plt.plot(X[:,0], Ypred_tf, lw=2, label='Iter#'+str(iter))
    plt.draw()
    print 'Plotting new fit'
    plt.show(block=False)

pdb.set_trace()

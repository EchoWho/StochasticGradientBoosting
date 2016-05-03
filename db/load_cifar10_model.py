#!/usr/bin/env python

import tensorflow as tf
import tensorpack
import tensorpack.examples.cifar10_resnet as cifar
import tensorpack.tfutils

import os
import numpy as np

from IPython import embed

def load_cifar_model(n, param_file, sess):
    """
    :n - Number of bottlenecks per channel size  in the res net
    :param_file -  model file (needs to correspond with the n) [i.e. cpkt file]
    :sess - tensorflow session
    :rtype - Returns a 'Model' object from *tensorpack* which has 
        the .gap, .probs for ouputs and access .image, .label for passing inputs
    """
    if not os.path.isfile(param_file):
        raise IOError('Cannot find file: {}'.format(param_file))

    model = cifar.Model(n=n)
    input_vars = model.get_input_vars()
    cost_var = model.get_cost(input_vars, is_training=False)


    saver = tf.train.Saver()
    saver.restore(sess, param_file)
    return model

def load_cifar_train_test():
    """
    :rtype - Returns tuple with train batches and test batches.
            d_train[0] is the first batch.
            d_train[0][0] corresponds to the images in the first batch
            d_train[0][1] corresponds to the labels in the first batch
    """
    ds_train = cifar.get_data('train')
    d_train = [d for d in ds_train.get_data()]
    ds_test = cifar.get_data('test')
    d_test = [d for d in ds_test.get_data()]
    return d_train, d_test

if __name__ == "__main__":
    """
    Example for how to load trained tensorpack model for cifar resnet
    """
    n = 5 # default value
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('n', default=n, 
        help='Number of bottlenecks for loading model (default n={})'.format(n))
    args = parser.parse_args()
    n = int(args.n)
    if n == 5:
        param_file = '/data/src/tensorpack/train_log/cifar10-resnet/model-195000'
    elif n == 18:
        param_file = '/data/src/tensorpack/train_log_18/cifar10-resnet/model-195000'
    else:
        raise Exception('No model known for n={}'.format(n))
    print('Running n = {} with model file {}'.format(n, param_file))
    sess = tf.Session()
    model = load_cifar_model(n, param_file, sess)
    d_train, d_test = load_cifar_train_test()
    batch_acc = np.zeros(len(d_test))

    for i,batch in enumerate(d_test):
        images = batch[0]
        labels = batch[1]
        # can also pass in labels with  model.label:labels
        gap, probs  = sess.run([model.gap, model.probs], feed_dict={model.image:images})
        ypred = probs.argmax(axis=1)
        batch_acc[i] = np.sum(ypred == labels)/float(len(labels))
    print('Overall accuracy: {:3.2f}%'.format(np.mean(batch_acc)*100.))

    embed()

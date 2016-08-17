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
    :rtype - Returns tuple with train data and test data.
            d_train[0] are the *images* in the train set
            d_train[1] are the *labels* in the train set
    """
    def convert_mini_to_full_batch(batches):
        images = np.vstack([batch[0] for batch in batches])
        labels = np.hstack([batch[1] for batch in batches])
        return images, labels

    ds_train = cifar.get_data('train')
    d_train = [d for d in ds_train.get_data()]

    ds_test = cifar.get_data('test')
    d_test = [d for d in ds_test.get_data()]
    return convert_mini_to_full_batch(d_train), convert_mini_to_full_batch(d_test)

if __name__ == "__main__":
    """
    Example for how to load trained tensorpack model for cifar resnet
    """
    n = 5  # default value
    out_file = '/data/data/processed_cifar_resnet.npz'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', default=n,
                        help='Number of bottlenecks for loading model (default n={})'.format(n))
    parser.add_argument('-s', default=out_file,
                        help='Location to save cifar data (default {})'.format(out_file))
    args = parser.parse_args()
    n = int(args.n)
    out_file = args.s
    if n == 5:
        param_file = '/data/src/tensorpack/train_log/cifar10-resnet/model-195000'
    elif n == 18:
        param_file = '/data/src/tensorpack/train_log_18/cifar10-resnet/model-195000'
    else:
        raise Exception('No model known for n={}'.format(n))
    print('Running n = {} with model file {}'.format(n, param_file))
    d_train, d_test = load_cifar_train_test()
    sess = tf.Session()
    model = load_cifar_model(n, param_file, sess)
    batch_acc = np.zeros(len(d_test))

    batch_size = 128
    train_images = d_train[0]
    labels_train = d_train[1]
    gap_train = []
    probs_train = []
    for si in range(0, train_images.shape[0], batch_size):
        si_end = min(si + batch_size, train_images.shape[0])
        gap_train_i, probs_train_i = sess.run([model.gap, model.probs],
                                              feed_dict={model.image: train_images[si:si_end]})
        gap_train.append(gap_train_i)
        probs_train.append(probs_train_i)
    gap_train = np.vstack(gap_train)
    probs_train = np.vstack(probs_train)

    test_images = d_test[0]
    labels_test = d_test[1]
    gap_test, probs_test = sess.run([model.gap, model.probs], feed_dict={model.image: test_images})
    # can also pass in labels with  model.label:labels
    ypred = probs_test.argmax(axis=1)
    print('Overall accuracy: {:3.2f}%'.format(np.sum(ypred == labels_test) / float(len(labels_test)) * 100.))

    def convert_to_one_hot(labels):
        n_samples = labels.shape[0]
        one_hot = np.zeros((n_samples, 10))
        one_hot[np.arange(n_samples), labels] = 1.
        return one_hot
    # before saving, convert labels to one-hot encoding
    labels_train = convert_to_one_hot(labels_train)
    labels_test = convert_to_one_hot(labels_test)

    im_train_feature = train_images.reshape((train_images.shape[0], -1))
    im_test_feature = test_images.reshape((test_images.shape[0], -1))
    np.savez(out_file,
             x_tra=gap_train, y_tra=labels_train, yp_tra=probs_train,
             x_test=gap_test, y_test=labels_test, yp_test=probs_test,
             im_train=im_train_feature, im_test=im_test_feature
             )
    print('Saved to: {}'.format(out_file))

    embed()

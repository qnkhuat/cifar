import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import time
import os
import random
import data_process as dp
from collections import Counter
import operator

_keep_rate = 1
_iter = 1
_lr = 0.0001


def create_placeholders(n_H, n_W, n_C, n_y):
    X = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X, Y


def prepare_params(X):
    m_train, n_H_train, n_W_train, n_C_train = X.shape
    X, Y = create_placeholders(n_H_train, n_W_train, n_C_train, 10)

    return X, Y


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def var_summary(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def conv(name, X_train, in_c, out_c, is_max_pool=False):
    name_scope = 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W' + name, [3, 3, in_c, out_c], initializer=tf.contrib.keras.initializers.he_normal())
            var_summary(W)
        with tf.name_scope('biases'):
            b = tf.get_variable('b' + name, [out_c], initializer=tf.constant_initializer(0.1))
            var_summary(b)

        Z = tf.nn.conv2d(X_train, W, strides=[1, 2, 2, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(Z, b)
        batch_norm = tf.contrib.layers.batch_norm(pre_activation, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                                  updates_collections=None)
        out = tf.nn.relu(batch_norm)

        if is_max_pool:
            out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return out


def fc(input,dropout,is_dropout=True):
    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(input)
        if is_dropout:
            output = tf.nn.dropout(output,dropout)
        output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
        output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


def fc_vgg(activation,out_c, dropout, is_dropout=True,is_activate=True):
    if is_activate:
        activate=tf.nn.relu
    else:
        activate=None

    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(activation)
        if is_dropout:
            output = tf.nn.dropout(output, dropout)
        output = tf.contrib.layers.fully_connected(output, out_c, activation_fn=activate)
    return output

def forward_prop(X_train, dropout):
    convs = conv('1', X_train, 3, 8)
    convs = conv('2', convs, 8, 16, is_max_pool=True)

    convs = conv('3', convs, 16, 32)
    convs = conv('4', convs, 32, 64, is_max_pool=True)

    convs = conv('5', convs, 64, 128)
    convs = conv('6', convs, 128, 256, is_max_pool=True)

    convs = conv('7', convs, 256, 512)
    convs = conv('8', convs, 512, 512, is_max_pool=True)

    convs = conv('9', convs, 512, 512)
    convs = conv('10', convs, 512, 512, is_max_pool=True)


    # output = fc(conv6, dropout)
    #
    # fc1 = fc_vgg(convs,4096,dropout)
    # fc2 = fc_vgg(fc1, 4096, dropout)

    fc3 = fc_vgg(convs, 10, dropout,is_dropout=False,is_activate=False)
    output=tf.nn.softmax(fc3)
    return output


def predict(Y, output):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy


def main():

    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = dp.loadData()
    X_train_origin,X_test_origin=dp.data_preprocessing(X_train_origin,X_test_origin)

    random_ex=np.random.permutation(100)
    X_test_origin=X_test_origin[random_ex]
    Y_test_origin=Y_test_origin[random_ex]

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    ops.reset_default_graph()

    X, Y = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)

    output = forward_prop(X, keep_prob)


    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)


        try:
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("Haha I found a checkpoint.")
        except:
            print("No checkpoint found.")

        for i in range(_iter):

            result = sess.run( output,feed_dict={X: X_test_origin, Y: Y_test_origin, keep_prob: 1.0})

            numberic_result=np.argmax(result,1)


            predicted=[]
            predicted.append([classes[i] for i in numberic_result])

            truth=[]
            numberic_truth=np.argmax(Y_test_origin,1)
            truth.append([classes[i] for i in numberic_truth])


            print('First 10 images')
            print("Result:")
            print(predicted[0][0:10])
            print("Truth:")
            print(truth[0][0:10])


            print("\n\nOccurrences")

            print("Result:")
            print(sorted(Counter(predicted[0]).items(),key=operator.itemgetter(0)))
            print("Truth:")
            print(sorted(Counter(truth[0]).items(),key=operator.itemgetter(0)))







if __name__ == "__main__":
    main()
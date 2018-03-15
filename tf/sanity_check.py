import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import time
import os
import random
import data_process as dp

_keep_rate = 0.5
_iter = 3000
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




def conv(name, X_train, in_c, out_c,strides=2, is_max_pool=False):
    name_scope= 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W'+name, [4, 4, in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope('biases'):
            b = tf.get_variable('b'+name,[out_c],initializer=tf.constant_initializer(0.1))

        Z = tf.nn.conv2d(X_train,W,strides=[1,strides,strides,1],padding='SAME')
        pre_activation = tf.nn.bias_add(Z,b)
        batch_norm=tf.contrib.layers.batch_norm(pre_activation,decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None)
        out=tf.nn.relu(batch_norm)

        if is_max_pool:
            out=tf.nn.max_pool(out,ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding='SAME')

    return out

def fc(activation,dropout,is_dropout=True):
    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(activation)
        if is_dropout:
            output = tf.nn.dropout(output,dropout)
        output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
        output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


def forward_prop(X_train,dropout):
    conv1=conv('1',X_train,3,8)
    conv2=conv('2',conv1,8,16,is_max_pool=True)

    conv3 = conv('3', conv2, 16, 32)
    conv4 = conv('4', conv3, 32, 64,is_max_pool=True)

    conv5 = conv('5', conv4, 64, 128)
    conv6 = conv('6', conv5, 128, 256, is_max_pool=True)

    output = fc(conv6,dropout)

    return output


def predict(Y,output):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy



def main():
    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = dp.loadData()
    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = X_train_origin[0:50], Y_train_origin[0:50], X_test_origin[0:20], Y_test_origin[0:20]

    ops.reset_default_graph()

    X, Y = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)

    output = forward_prop(X, keep_prob)
    cost = compute_cost(output, Y)

    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate=float(_lr)).minimize(cost)

    with tf.name_scope('accuracy'):
        accuracy=predict(Y,output)


    init = tf.global_variables_initializer()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)

        for i in range(_iter):
          start_time = time.time()
          _, temp_cost = sess.run([optimizer, cost],feed_dict={X: X_train_origin, Y: Y_train_origin, keep_prob: _keep_rate})

          end_time = time.time()
          total_time = end_time - start_time
          if i % 10 == 0:

              train_accuracy = sess.run(accuracy,feed_dict={X:X_train_origin,Y:Y_train_origin,keep_prob:_keep_rate})

              test_accuracy = sess.run(accuracy,feed_dict={X:X_test_origin,Y:Y_test_origin,keep_prob:1.0})

              # save data to txt
              print("cost after {} iters : {} in {} each with train accuracy = {} and test accuracy = {} ".format(i,
                                                                                                                  temp_cost,
                                                                                                                  total_time,
                                                                                                                  train_accuracy,
                                                                                                                  test_accuracy))


if __name__ == "__main__":
    main()

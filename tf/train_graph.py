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
_iter = 10000
_lr = 0.0001


def create_placeholders(n_H, n_W, n_C, n_y):
    X_train = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y_train = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X_train, Y_train


def prepare_params(X_train):
    m_train, n_H_train, n_W_train, n_C_train = X_train.shape
    X_train, Y_train = create_placeholders(n_H_train, n_W_train, n_C_train, 10)

    return X_train, Y_train


def compute_cost(Z3, Y):
    with tf.name_scope('cost'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
        tf.summary.scalar('cost',cost)
    return cost


def conv(W_name, X_train, in_c, out_c,strides=2, is_max_pool=False):
    name_scope= 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        W = tf.get_variable(W_name, [4, 4, in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
        Z=tf.nn.conv2d(X_train,W,strides=[1,strides,strides,1],padding='SAME')
        batch_norm=tf.contrib.layers.batch_norm(Z,decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None)
        out=tf.nn.relu(batch_norm)

        if is_max_pool:
            out=tf.nn.max_pool(out,ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding='SAME')

    return out



def fc(input,dropout,is_dropout=True):
    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(input)
        if is_dropout:
            output = tf.nn.dropout(output,dropout)
        output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
        output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


def forward_prop(X_train,dropout):
    conv1=conv('W1',X_train,3,8)
    conv2=conv('W2',conv1,8,16,is_max_pool=True)

    conv3 = conv('W3', conv2, 16, 32)
    conv4 = conv('W4', conv3, 32, 64,is_max_pool=True)

    conv5 = conv('W5', conv4, 64, 128)
    conv6 = conv('W6', conv5, 128, 256, is_max_pool=True)

    output = fc(conv6,dropout)

    return output


def predict(X_train_origin, Y_train_origin, X_test_origin, Y_test_origin, output, X_train, Y_train, keep_prob):
    predict_op = tf.argmax(output, 1)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(predict_op, tf.argmax(Y_train, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_accuracy = accuracy.eval({X_train: X_train_origin, Y_train: Y_train_origin, keep_prob: 1.0})
    test_accuracy = accuracy.eval({X_train: X_test_origin, Y_train: Y_test_origin, keep_prob: 1.0})

    tf.summary.scalar('train_accuracy',train_accuracy)
    tf.summary.scalar('test_accuracy',test_accuracy)

    return train_accuracy, test_accuracy


def main():
    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = dp.loadData()

    ops.reset_default_graph()

    X_train, Y_train = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)

    output = forward_prop(X_train, keep_prob)
    cost = compute_cost(output, Y_train)
    optimizer = tf.train.AdamOptimizer(learning_rate=_lr).minimize(cost)

    m = X_train_origin.shape[0]
    minibatch_size = 64

    # load data from txt
    cache_files_name=['data/costs.txt', 'data/trains.txt', 'data/tests.txt']
    dp.ensure_dir(cache_files_name)
    costs, trains, tests = dp.load_txt(cache_files_name)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('graph/8',sess.graph)

        try:
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            print("No checkpoint found")

        for i in range(_iter):
            start_time = time.time()

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = dp.random_mini_batches(X_train_origin, Y_train_origin, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X=dp.data_augmentation(minibatch_X)
                summary,_, temp_cost = sess.run([merged,optimizer, cost],feed_dict={X_train: minibatch_X, Y_train: minibatch_Y, keep_prob: _keep_rate})
                minibatch_cost += temp_cost / num_minibatches
                writer.add_summary(summary, i)


            costs = np.append(costs, minibatch_cost)

            end_time = time.time()
            total_time = end_time - start_time

            if i % 10 == 0:
                saver.save(sess, "./checkpoint/model.ckpt", global_step=1)

                train_accuracy, test_accuracy = predict(X_train_origin, Y_train_origin, X_test_origin, Y_test_origin,
                                                        output, X_train, Y_train, keep_prob)


                trains, tests = dp.append_data(trains, tests, train_accuracy, test_accuracy)

                # save data to txt
                dp.save_txt(costs, trains, tests,cache_files_name)
                print("cost after {} iters : {} in {} each with train accuracy = {} and test accuracy = {} ".format(i,
                                                                                                                    minibatch_cost,
                                                                                                                    total_time,
                                                                                                                    train_accuracy,
                                                                                                                    test_accuracy))


if __name__ == "__main__":
    main()

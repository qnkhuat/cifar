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
_iter = 500
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



def conv(name, X_train, in_c, out_c,strides=2, is_max_pool=False):
    name_scope= 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W'+name, [4, 4, in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
            var_summary(W)
        with tf.name_scope('biases'):
            b = tf.get_variable('b'+name,[out_c],initializer=tf.constant_initializer(0.1))
            var_summary(b)

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
    lrs = [0.01,0.001,0.0001,0.05,0.005,0.0005]
    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = dp.loadData()

    ops.reset_default_graph()

    X, Y = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)

    output = forward_prop(X, keep_prob)
    cost = compute_cost(output, Y)

    m = X_train_origin.shape[0]
    minibatch_size = 64

    for lr in lrs:
        _lr = str(lr)
        cache_files_name = ['data/costs_'+_lr+'.txt','data/trains_'+_lr+'.txt','data/tests_'+_lr+'.txt']
        dp.ensure_dir(cache_files_name)
        checkpoint_folder_name = './checkpoints/checkpoint_' + _lr + '/'
        with tf.name_scope('train'):
            optimizer = tf.train.AdamOptimizer(learning_rate=float(_lr)).minimize(cost)


        with tf.name_scope('accuracy'):
            accuracy=predict(Y,output)
        tf.summary.scalar('accuracy',accuracy)

        # load data from txt
        dp.ensure_dir(cache_files_name)
        costs, trains, tests = dp.load_txt(cache_files_name)

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()



        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            sess.run(init)
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter('graph/3' + '/train',
                                          sess.graph)
            test_writer = tf.summary.FileWriter('graph/3' + '/test')

            try:

                ckpt = tf.train.get_checkpoint_state(checkpoint_folder_name)
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
                    _, temp_cost = sess.run([optimizer, cost],feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: _keep_rate})
                    minibatch_cost += temp_cost / num_minibatches

                costs = np.append(costs, minibatch_cost)

                end_time = time.time()
                total_time = end_time - start_time

                if i % 10 == 0:

                    summary , train_accuracy = sess.run([merged,accuracy],feed_dict={X:X_train_origin,Y:Y_train_origin,keep_prob:_keep_rate})
                    train_writer.add_summary(summary, i)

                    summary , test_accuracy = sess.run([merged,accuracy],feed_dict={X:X_test_origin,Y:Y_test_origin,keep_prob:1.0})
                    test_writer.add_summary(summary, i)


                    trains, tests = dp.append_data(trains, tests, train_accuracy, test_accuracy)

                    # save data to txt
                    dp.save_txt(costs, trains, tests, cache_files_name)
                    print("cost after {} iters : {} in {} each with train accuracy = {} and test accuracy = {} ".format(i,
                                                                                                                        minibatch_cost,
                                                                                                                        total_time,
                                                                                                                        train_accuracy,
                                                                                                                        test_accuracy))

                    saver.save(sess, checkpoint_folder_name+ "model.ckpt", global_step=1)


if __name__ == "__main__":
    main()

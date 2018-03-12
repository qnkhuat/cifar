import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import math
import time
import os
import random
files_train = {"../train/data/data_batch_1", "../train/data/data_batch_2", "../train/data/data_batch_3",
               "../train/data/data_batch_4", "../train/data/data_batch_5"}
files_test = {"../train/data/test_batch"}

_keep_rate = 0.5
_iter = 10000
_lr = 0.0001


def _random_crop(batch, crop_shape, padding=None):
    oshape = np.shape(batch[0])

    if padding:
        oshape = (oshape[0] + 2 * padding, oshape[1] + 2 * padding)
    new_batch = []
    npad = ((padding, padding), (padding, padding), (0, 0))
    for i in range(len(batch)):
        new_batch.append(batch[i])
        if padding:
            new_batch[i] = np.lib.pad(batch[i], pad_width=npad,
                                      mode='constant', constant_values=0)
        nh = random.randint(0, oshape[0] - crop_shape[0])
        nw = random.randint(0, oshape[1] - crop_shape[1])
        new_batch[i] = new_batch[i][nh:nh + crop_shape[0],
                       nw:nw + crop_shape[1]]
    return new_batch


def _random_flip_leftright(batch):
    for i in range(len(batch)):
        if bool(random.getrandbits(1)):
            batch[i] = np.fliplr(batch[i])
    return batch

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    return batch

def loadData():
    X_train = np.zeros((len(files_train) * 10000, 3072))
    Y_train = np.zeros((len(files_train) * 10000))

    X_test = np.zeros((len(files_test) * 10000, 3072))
    Y_test = np.zeros((len(files_test) * 10000))

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    for index, file in enumerate(files_train):
        A = unpickle(file)
        start = index * 10000
        end = (index + 1) * 10000
        X_train[start:end, :] = A[b'data']
        Y_train[start:end] = A[b'labels']

    for index, file in enumerate(files_test):
        A = unpickle(file)
        start = index * 10000
        end = (index + 1) * 10000
        X_test[start:end, :] = A[b'data']
        Y_test[start:end] = A[b'labels']

    X_train = X_train / 255
    X_test = X_test / 255

    # reshape and one hot data
    X_train = X_train.reshape((len(files_train) * 10000, 3, 32, 32))
    Y_train = np.eye(10)[Y_train.astype(int)]

    X_test = X_test.reshape((len(files_test) * 10000, 3, 32, 32))
    Y_test = np.eye(10)[Y_test.astype(int)]

    return X_train.transpose([0, 2, 3, 1]), Y_train, X_test.transpose([0, 2, 3, 1]), Y_test


def create_placeholders(n_H, n_W, n_C, n_y):
    X_train = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y_train = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X_train, Y_train


def prepare_params(X_train):
    m_train, n_H_train, n_W_train, n_C_train = X_train.shape
    X_train, Y_train = create_placeholders(n_H_train, n_W_train, n_C_train, 10)

    return X_train, Y_train


def compute_cost(Z3, Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    return cost


def conv(W_name, X_train, in_c, out_c,strides=2, is_max_pool=False):
    W = tf.get_variable(W_name, [4, 4, in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
    Z=tf.nn.conv2d(X_train,W,strides=[1,strides,strides,1],padding='SAME')
    batch_norm=tf.contrib.layers.batch_norm(Z,decay=0.9, center=True, scale=True, epsilon=1e-3, updates_collections=None)
    out=tf.nn.relu(batch_norm)

    if is_max_pool:
        out=tf.nn.max_pool(out,ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding='SAME')

    return out

def forward_prop(X_train,dropout):
    conv1=conv('W1',X_train,3,8)
    conv2=conv('W2',conv1,8,16,is_max_pool=True)

    conv3 = conv('W3', conv2, 16, 32)
    conv4 = conv('W4', conv3, 32, 64,is_max_pool=True)

    conv5 = conv('W5', conv4, 64, 128)
    conv6 = conv('W6', conv5, 128, 256, is_max_pool=True)

    output = fc(conv6,dropout)

    return output

def fc(input,dropout,is_dropout=True):
    output = tf.contrib.layers.flatten(input)
    if is_dropout:
        output = tf.nn.dropout(output,dropout)
    output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
    output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


# GRADED FUNCTION: random_mini_batches

def random_mini_batches(X, Y, mini_batch_size=64):
    m = X.shape[0]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_Y = Y[permutation, :]
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        ### START CODE HERE ### (approx. 2 lines)
        mini_batch_X = shuffled_X[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size, :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        ### START CODE HERE ### (approx. 2 lines)
        # end = m - mini_batch_size * math.floor(m / mini_batch_size)
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size:, :]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size:, :]
        ### END CODE HERE ###
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def append_data(trains, tests, train_accuracy, test_accuracy):
    trains = np.append(trains, train_accuracy)
    tests = np.append(tests, test_accuracy)
    return trains, tests



def ensure_dir(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            with open(dir, 'w'): pass


def load_txt(cost_dir, train_dir, test_dir):
    costs = np.loadtxt(cost_dir, dtype=float)
    trains = np.loadtxt(train_dir, dtype=float)
    tests = np.loadtxt(test_dir, dtype=float)
    return costs, trains, tests


def save_txt(costs, trains, tests, cost_dir, train_dir, test_dir):
    np.savetxt(cost_dir, costs, fmt='%1.16f')
    np.savetxt(train_dir, trains, fmt='%1.16f')
    np.savetxt(test_dir, tests, fmt='%1.16f')


def predict(X_train_origin, Y_train_origin, X_test_origin, Y_test_origin, output, X_train, Y_train, keep_prob):
    predict_op = tf.argmax(output, 1)
    correct_prediction = tf.equal(predict_op, tf.argmax(Y_train, 1))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    train_accuracy = accuracy.eval({X_train: X_train_origin, Y_train: Y_train_origin, keep_prob: 1.0})
    test_accuracy = accuracy.eval({X_train: X_test_origin, Y_train: Y_test_origin, keep_prob: 1.0})

    return train_accuracy, test_accuracy


def main():
    X_train_origin, Y_train_origin, X_test_origin, Y_test_origin = loadData()

    ops.reset_default_graph()

    X_train, Y_train = prepare_params(X_train_origin)

    keep_prob = tf.placeholder(tf.float32)

    # output = forward_prop(X_train, weights, keep_prob)
    output = forward_prop(X_train, keep_prob)
    cost = compute_cost(output, Y_train)
    optimizer = tf.train.AdamOptimizer(learning_rate=_lr).minimize(cost)

    m = X_train_origin.shape[0]
    minibatch_size = 64

    # load data from txt
    ensure_dir(['data/costs.txt', 'data/trains.txt', 'data/tests.txt'])
    costs, trains, tests = load_txt('data/costs.txt', 'data/trains.txt', 'data/tests.txt')

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)
        try:
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            print("No checkpoint found")

        for i in range(_iter):
            start_time = time.time()

            minibatch_cost = 0
            num_minibatches = int(m / minibatch_size)
            minibatches = random_mini_batches(X_train_origin, Y_train_origin, minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X=data_augmentation(minibatch_X)
                _, temp_cost = sess.run([optimizer, cost],feed_dict={X_train: minibatch_X, Y_train: minibatch_Y, keep_prob: _keep_rate})
                minibatch_cost += temp_cost / num_minibatches

            costs = np.append(costs, minibatch_cost)

            end_time = time.time()
            total_time = end_time - start_time

            if i % 10 == 0:
                saver.save(sess, "./checkpoint/model.ckpt", global_step=1)

                train_accuracy, test_accuracy = predict(X_train_origin, Y_train_origin, X_test_origin, Y_test_origin,
                                                        output, X_train, Y_train, keep_prob)

                trains, tests = append_data(trains, tests, train_accuracy, test_accuracy)

                # save data to txt
                save_txt(costs, trains, tests, 'data/costs.txt', 'data/trains.txt', 'data/tests.txt')
                print("cost after {} iters : {} in {} each with train accuracy = {} and test accuracy = {} ".format(i,
                                                                                                                    minibatch_cost,
                                                                                                                    total_time,
                                                                                                                    train_accuracy,
                                                                                                                    test_accuracy))

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()
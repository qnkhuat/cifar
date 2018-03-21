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

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def loadData():
    X_train = np.zeros((len(files_train) * 10000, 3072))
    Y_train = np.zeros((len(files_train) * 10000))

    X_test = np.zeros((len(files_test) * 10000, 3072))
    Y_test = np.zeros((len(files_test) * 10000))

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


    # reshape and one hot data
    X_train = X_train.reshape((len(files_train) * 10000, 3, 32, 32))
    Y_train = np.eye(10)[Y_train.astype(int)]

    X_test = X_test.reshape((len(files_test) * 10000, 3, 32, 32))
    Y_test = np.eye(10)[Y_test.astype(int)]

    return X_train.transpose([0, 2, 3, 1]), Y_train, X_test.transpose([0, 2, 3, 1]), Y_test

def data_preprocessing(x_train,x_test):

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train[:,:,:,0] = (x_train[:,:,:,0] - np.mean(x_train[:,:,:,0])) / np.std(x_train[:,:,:,0])
    x_train[:,:,:,1] = (x_train[:,:,:,1] - np.mean(x_train[:,:,:,1])) / np.std(x_train[:,:,:,1])
    x_train[:,:,:,2] = (x_train[:,:,:,2] - np.mean(x_train[:,:,:,2])) / np.std(x_train[:,:,:,2])

    x_test[:,:,:,0] = (x_test[:,:,:,0] - np.mean(x_test[:,:,:,0])) / np.std(x_test[:,:,:,0])
    x_test[:,:,:,1] = (x_test[:,:,:,1] - np.mean(x_test[:,:,:,1])) / np.std(x_test[:,:,:,1])
    x_test[:,:,:,2] = (x_test[:,:,:,2] - np.mean(x_test[:,:,:,2])) / np.std(x_test[:,:,:,2])

    return x_train, x_test

def data_augmentation(batch):
    batch = _random_flip_leftright(batch)
    batch = _random_crop(batch, [32,32], 4)
    return batch


def append_data(trains, tests, train_accuracy, test_accuracy):
    trains = np.append(trains, train_accuracy)
    tests = np.append(tests, test_accuracy)
    return trains, tests


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

def ensure_dir(dirs):
    for dir in dirs:
        if not os.path.exists(dir):
            with open(dir, 'w'): pass


def load_txt(cache_files_name):
    costs = np.loadtxt(cache_files_name[0], dtype=float)
    trains = np.loadtxt(cache_files_name[1], dtype=float)
    tests = np.loadtxt(cache_files_name[2], dtype=float)
    return costs, trains, tests


def save_txt(costs, trains, tests, cache_files_name):
    np.savetxt(cache_files_name[0], costs, fmt='%1.16f')
    np.savetxt(cache_files_name[1], trains, fmt='%1.16f')
    np.savetxt(cache_files_name[2], tests, fmt='%1.16f')

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
from train import *
_keep_rate = 1
_iter = 1
_lr = 0.0001



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
            print(random_ex[0:10])
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
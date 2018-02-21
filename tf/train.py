import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
files_train={"../train/data/data_batch_1","../train/data/data_batch_2","../train/data/data_batch_3"}
files_test={"../train/data/test_batch"}

def loadData():
    X_train = np.zeros((len(files_train)*10000,3072))
    Y_train = np.zeros((len(files_train)*10000))

    X_test = np.zeros((len(files_test) * 10000, 3072))
    Y_test = np.zeros((len(files_test) * 10000))

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    for index,file in enumerate(files_train):

        A=unpickle(file)
        start = index*10000
        end = (index+1)*10000
        X_train[start:end,:] = A[b'data']
        Y_train[start:end] = A[b'labels']

    for index,file in enumerate(files_test):

        A=unpickle(file)
        start = index*10000
        end = (index+1)*10000
        X_test[start:end,:] = A[b'data']
        Y_test[start:end] = A[b'labels']


    X_train=X_train/255
    X_test=X_test/255
    #reshape and one hot data
    X_train = X_train.reshape((len(files_train)*10000,32,32,3))
    Y_train = np.eye(10)[Y_train.astype(int)]

    X_test = X_test.reshape((len(files_test) * 10000, 32, 32, 3))
    Y_test = np.eye(10)[Y_test.astype(int)]

    return X_train , Y_train, X_test,Y_test


def create_placeholders(n_H,n_W,n_C,n_y):
    X_train = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y_train = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X_train,Y_train

def prepare_params(X_train,X_test):
    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer())#shape [filter_height, filter_width, in_channels, out_channels]
    W2 = tf.get_variable("W2", [4, 4, 8, 16], initializer=tf.contrib.layers.xavier_initializer())#shape [filter_height, filter_width, in_channels, out_channels]

    m_train,n_H_train,n_W_train,n_C_train=X_train.shape
    X_train,Y_train=create_placeholders(n_H_train,n_W_train,n_C_train,10)


    m_test,n_H_test,n_W_test,n_C_test=X_test.shape
    X_test, Y_test = create_placeholders(n_H_test, n_W_test, n_C_test, 10)


    return X_train,Y_train,X_test,Y_test,W1,W2


def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3, labels=Y))
    print(Z3)
    return cost

def forward_prop(X_train,W1,W2):


    Z1=tf.nn.conv2d(X_train,W1,strides=[1,1,1,1],padding='SAME')
    A1=tf.nn.relu(Z1)
    P1=tf.nn.max_pool(A1,ksize = [1,8,8,1], strides = [1,8,8,1], padding = 'SAME')

    Z2=tf.nn.conv2d(P1,W2,strides=[1,2,2,1],padding="SAME")
    A2=tf.nn.relu(Z2)
    P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides = [1,4,4,1], padding = 'SAME')

    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2, 10, activation_fn=None)
    return Z3



def main():

    X_train_origin, Y_train_origin, X_test_origin,Y_test_origin =loadData()

    ops.reset_default_graph()

    X_train, Y_train, X_test, Y_test, W1, W2 = prepare_params(X_train_origin, X_test_origin)

    m_train,n_H_train,n_W_train,n_C_train=X_train.shape

    X,Y=create_placeholders(n_H_train,n_W_train,n_C_train,10)

    Z3 = forward_prop(X_train, W1, W2)
    cost = compute_cost(Z3, Y_train)
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(cost)


    init = tf.global_variables_initializer()
    # saver = tf.train.Saver()
    with tf.Session() as sess:
        # saver.restore(sess, "/tmp/model.ckpt")
        sess.run(init)
        for i in range(1):

            _, costs = sess.run([optimizer, cost], feed_dict={X_train : X_train_origin,Y_train: Y_train_origin})
            print(costs)

        # saved=saver.save(sess,"tmp/model.ckpt")
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y_train, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X_train : X_train_origin,Y_train: Y_train_origin})
        test_accuracy = accuracy.eval({X_train : X_test_origin,Y_train: Y_test_origin})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


if __name__ == "__main__":
    main()

import numpy as np
import argparse
import tensorflow as tf
from tensorflow.python.framework import ops

parser = argparse.ArgumentParser()
parser.add_argument('index')
args=parser.parse_args()


_lr = 0.0001
_keep_rate = 0.5

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_path="../train/data/data_batch_1"

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

A=unpickle(data_path)
X=A[b'data']
Y=A[b'labels']

idx=int(args.index)#index of images want to visualize
X_test=X[idx]
Y_test=Y[idx]
Y_test = np.eye(10)[Y_test]
Y_test= np.expand_dims(Y_test,axis=0)


#unflat
X_test=X_test.reshape(3,32,32)

#tranpose
X_test=X_test.transpose([1, 2, 0])
X_test=np.expand_dims(X_test,axis=0)


def conv(name, X_train, in_c, out_c, strides=2, is_max_pool=False):
    name_scope = 'conv_max' if is_max_pool else 'conv'

    with tf.name_scope(name_scope):
        with tf.name_scope('weights'):
            W = tf.get_variable('W' + name, [4, 4, in_c, out_c], initializer=tf.contrib.layers.xavier_initializer())
        with tf.name_scope('biases'):
            b = tf.get_variable('b' + name, [out_c], initializer=tf.constant_initializer(0.1))

        Z = tf.nn.conv2d(X_train, W, strides=[1, strides, strides, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(Z, b)
        batch_norm = tf.contrib.layers.batch_norm(pre_activation, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                                  updates_collections=None)
        out = tf.nn.relu(batch_norm)

        if is_max_pool:
            out = tf.nn.max_pool(out, ksize=[1, 2, 2, 1], strides=[1, strides, strides, 1], padding='SAME')

    return out


def fc(activation, dropout, is_dropout=True):
    with tf.name_scope('fc'):
        output = tf.contrib.layers.flatten(activation)
        if is_dropout:
            output = tf.nn.dropout(output, dropout)
        output = tf.contrib.layers.fully_connected(output, 2048, activation_fn=None)
        output = tf.contrib.layers.fully_connected(output, 1024, activation_fn=None)
        output = tf.contrib.layers.fully_connected(output, 10, activation_fn=None)
    return output


def create_placeholders(n_H, n_W, n_C, n_y):
    X = tf.placeholder(shape=[None, n_H, n_W, n_C], dtype=tf.float32)
    Y = tf.placeholder(shape=[None, n_y], dtype=tf.float32)

    return X, Y


def prepare_params(X):
    m, n_H_train, n_W_train, n_C_train = X.shape
    X, Y = create_placeholders(n_H_train, n_W_train, n_C_train, 10)

    return X, Y
def forward_prop(X_train, dropout):
    convs = conv('1', X_train, 3, 8)
    convs = conv('2', convs, 8, 16, is_max_pool=True)

    convs = conv('3', convs, 16, 32)
    convs = conv('4', convs, 32, 64, is_max_pool=True)

    convs = conv('5', convs, 64, 128)
    convs = conv('6', convs, 128, 256, is_max_pool=True)

    convs = conv('7', convs, 256, 512)
    convs = conv('8', convs, 512, 512, is_max_pool=True)

    output = fc(convs, dropout)

    return output

def predict(Y, output):
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    return accuracy

def main():
    ops.reset_default_graph()

    X, Y = prepare_params(X_test)

    keep_prob = tf.placeholder(tf.float32)

    output = forward_prop(X, keep_prob)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        sess.run(init)

        try:
            ckpt = tf.train.get_checkpoint_state('./checkpoint/')
            saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            print("No checkpoint found")

        y = sess.run(output,feed_dict={X:X_test,Y:Y_test, keep_prob: _keep_rate})
        
        print(classes[np.argmax(y)])



if __name__ == '__main__':
    main()
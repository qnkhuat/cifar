import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('index')
args=parser.parse_args()


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
X=X[idx]
Y=Y[idx]

#unflat
X=X.reshape(3,32,32)

#tranpose
X=X.transpose([1, 2, 0])


plt.imshow(X)
plt.title(classes[Y])
plt.show()

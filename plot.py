import matplotlib.pyplot as plt
import numpy as np


#load files
cost=np.loadtxt('tf/data/costs.txt')
test=np.loadtxt('tf/data/tests.txt')
train=np.loadtxt('tf/data/trains.txt')


plt.figure()
plt.plot(cost)
plt.figure()
plt.plot(test,c='r')
plt.plot(train,c='b')

plt.show()

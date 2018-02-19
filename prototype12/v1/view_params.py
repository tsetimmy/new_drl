import pickle
import numpy as np

import pylab
from matplotlib import pyplot as plt
import sys
sys.path.append('../..')
from utils import dispims

'''
size = 'small_a'

data = pickle.load( open( "model_params_"+size+".p", "rb" ) )
conv1 = data[2]
conv2 = data[4]
recon_x = data[-2][0]
recon_y = data[-2][1]
batch = data[-1]
'''

data = pickle.load( open( "recons.p", "rb" ) )
conv1 = data[0]
conv2 = data[1]
batch = data[-3]##
recon_x = data[-2]##
recon_y = data[-1]##
x = batch[0]
y = batch[1]


'''
tmp = []
for i in range(len(batch)):
    tmp.append(batch[i][np.newaxis, ...])
batch = np.concatenate(tmp, axis=0)
'''

'''
for i in range(len(recon_x)):
    plt.subplot(3,4,1)
    plt.imshow(recon_x[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,2)
    plt.imshow(recon_x[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,3)
    plt.imshow(recon_x[i, :, :, 2], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,4)
    plt.imshow(recon_x[i, :, :, 3], interpolation='nearest', cmap='gray')

    plt.subplot(3,4,5)
    plt.imshow(recon_y[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,6)
    plt.imshow(recon_y[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,7)
    plt.imshow(recon_y[i, :, :, 2], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,8)
    plt.imshow(recon_y[i, :, :, 3], interpolation='nearest', cmap='gray')

    plt.subplot(3,4,9)
    plt.imshow(batch[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,10)
    plt.imshow(batch[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,11)
    plt.imshow(batch[i, :, :, 2], interpolation='nearest', cmap='gray')
    plt.subplot(3,4,12)
    plt.imshow(batch[i, :, :, 3], interpolation='nearest', cmap='gray')

    plt.show()
'''
    

'''
for i in range(len(recon_x)):
    plt.subplot(4,2,1)
    plt.imshow(recon_x[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,2)
    plt.imshow(recon_x[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,3)
    plt.imshow(recon_y[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,4)
    plt.imshow(recon_y[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,5)
    plt.imshow(x[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,6)
    plt.imshow(x[i, :, :, 1], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,7)
    plt.imshow(y[i, :, :, 0], interpolation='nearest', cmap='gray')
    plt.subplot(4,2,8)
    plt.imshow(y[i, :, :, 1], interpolation='nearest', cmap='gray')


    plt.show()
'''

print conv1.shape
conv1 = conv1.reshape(-1, 32)
conv2 = conv2.reshape(-1, 32)

conv1 = conv1[:121, :]
conv2 = conv2[:121, :]


pylab.clf()
#pylab.subplot(1,2,1)
dispims(conv1,11,11,2)
pylab.axis('off')

pylab.clf()
#pylab.subplot(1,2,1)
dispims(conv2,11,11,2)
pylab.axis('off')

print conv1.shape
print conv2.shape


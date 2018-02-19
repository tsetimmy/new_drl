import pickle
import numpy as np

import pylab
from matplotlib import pyplot as plt

def dispims(M, height, width, border=0, bordercolor=0.0, **kwargs):
    """ Display the columns of matrix M in a montage. """
    numimages = M.shape[1]
    n0 = np.int(np.ceil(np.sqrt(numimages)))
    n1 = np.int(np.ceil(np.sqrt(numimages)))
    im = bordercolor*\
         np.ones(((height+border)*n1+border,(width+border)*n0+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[j*(height+border)+border:(j+1)*(height+border)+border,\
                   i*(width+border)+border:(i+1)*(width+border)+border] = \
                np.vstack((\
                  np.hstack((np.reshape(M[:,i*n1+j],(width,height)).T,\
                         bordercolor*np.ones((height,border),dtype=float))),\
                  bordercolor*np.ones((border,width+border),dtype=float)\
                  ))
    plt.imshow(im.T,cmap=pylab.cm.gray,interpolation='nearest', **kwargs)
    #pylab.show()

def reshape_filters(filters):
    filters = filters.reshape(-1, filters.shape[-1])
    shape = int(np.sqrt(filters.shape[0]))
    return filters, shape


data = pickle.load( open( "model_params_breakout_cross_correlation.p", "rb" ) )
filters = data[4]
filters, shape = reshape_filters(filters)

plt.subplot(1,4,1)
dispims(filters,shape,shape,2)
plt.axis('off')

data = pickle.load( open( "model_params_breakout_cross_correlation_conv.p", "rb" ) )
filters = data[0]
filters, shape = reshape_filters(filters)

plt.subplot(1,4,2)
dispims(filters,shape,shape,2)
plt.axis('off')

data = pickle.load( open( "model_params_breakout_transformation.p", "rb" ) )
filters = data[4]
filters, shape = reshape_filters(filters)

plt.subplot(1,4,3)
dispims(filters,shape,shape,2)
plt.axis('off')

data = pickle.load( open( "model_params_breakout_transformation_conv.p", "rb" ) )
filters = data[0]
filters, shape = reshape_filters(filters)

plt.subplot(1,4,4)
dispims(filters,shape,shape,2)
plt.axis('off')


plt.show()

'''
d = [recon_x, recon_y, batch]
for i in range(len(recon_x)):
    count = 1
    for j in range(len(d)):
        for k in range(d[j].shape[-1]):
            plt.subplot(len(d),len(recon_x),count)
            plt.imshow(d[j][i, :, :, k], interpolation='nearest', cmap='gray')
            count += 1
    plt.show()
'''

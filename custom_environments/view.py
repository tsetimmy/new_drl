import pickle
from matplotlib import pyplot as plt




frame = pickle.load( open( "frame.p", "rb" ) )
plt.imshow(frame[-1], cmap='gray')
plt.show()



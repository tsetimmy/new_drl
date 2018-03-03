import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from experiments_data.plot import moving_average, get_rewards, average

ddpg = ['ddpg_test_1.txt', 'ddpg_test_2.txt', 'ddpg_test_3.txt', 'ddpg_test_4.txt', 'ddpg_test_5.txt']
ddpg_transfer = ['ddpg_transfer_1.txt', 'ddpg_transfer_2.txt', 'ddpg_transfer_3.txt', 'ddpg_transfer_4.txt', 'ddpg_transfer_5.txt']
jointgan = ['jointgan_test_1.txt', 'jointgan_test_2.txt', 'jointgan_test_3.txt', 'jointgan_test_4.txt', 'jointgan_test_5.txt']
jointgan_transfer = ['jointgan_transfer_1.txt', 'jointgan_transfer_2.txt', 'jointgan_transfer_3.txt', 'jointgan_transfer_4.txt', 'jointgan_transfer_5.txt']
jointgated = ['jointgated_test_1.txt', 'jointgated_test_2.txt', 'jointgated_test_3.txt', 'jointgated_test_4.txt', 'jointgated_test_5.txt']
jointgated_transfer = ['jointgated_transfer_1.txt', 'jointgated_transfer_2.txt', 'jointgated_transfer_3.txt', 'jointgated_transfer_4.txt', 'jointgated_transfer_5.txt'] 

replicate = ['replicate1.txt', 'replicate2.txt', 'replicate3.txt', 'replicate4.txt', 'replicate5.txt']

data = [ddpg, ddpg_transfer, jointgan, jointgan_transfer, jointgated, jointgated_transfer, replicate]

handles = []
for i in range(len(data)):
    y = np.cumsum(np.array(average(data[i])))
    x = np.arange(len(y))
    handle, = plt.plot(x, y, label=data[i][0].split('.')[0][:-2])
    handles.append(handle)
plt.legend(handles=handles)
plt.grid()
plt.show()


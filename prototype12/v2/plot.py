import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def moving_average(data, N):
    avg = []
    std = []
    for i in range(len(data) - N + 1):
        avg.append(np.average(data[i:i+N]))
        std.append(np.std(data[i:i+N]))
    return np.asarray(avg), np.asarray(std)

def average(scores, length):
    ret = []
    cum = 0.
    for i in range(len(scores)):
        cum += scores[i]
        if i % length == 0:
            ret.append(cum / float(length))
            cum = 0.
    return ret

filepath = 'tmp.txt'
rewards = []
rewards2 = []
with open(filepath) as fp:
    line = fp.readline()
    while line:
        if 'recon_loss' in line:
            l = line.strip().split(' ')
            rewards.append(float(l[3]))
            rewards2.append(float(l[5]))
        line = fp.readline()

#rewards, _ = moving_average(rewards, 100)
#ax = sns.regplot(x=np.arange(len(rewards)), y=np.array(rewards))
plt.scatter(np.arange(len(rewards)), rewards)
plt.scatter(np.arange(len(rewards2)), rewards2)
plt.grid()
plt.show()
exit()


import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from experiments_data.plot import moving_average, get_rewards

def average(fnames, truncated=False):
    rewards = []
    for fname in fnames:
        rewards.append(get_rewards(fname))

    maxlen = len(rewards[0])
    minlen = len(rewards[0])
    for i in range(len(rewards)):
        maxlen = max(maxlen, len(rewards[i]))
        minlen = min(minlen, len(rewards[i]))

    total = [0.] * maxlen
    divisor = [0.] * maxlen

    for i in range(len(rewards)):
        for j in range(len(rewards[i])):
            total[j] += rewards[i][j]
            divisor[j] += 1.

    for j in range(len(total)):
        total[j] /= divisor[j]

    if truncated:
        total = total[:minlen]

    return total

    


Ant = ['Ant_gated_1.txt', 'Ant_gated_2.txt', 'Ant_gated_3.txt', 'Ant_gated_4.txt', 'Ant_gated_5.txt']
HalfCheetah = ['HalfCheetah_gated_1.txt', 'HalfCheetah_gated_2.txt', 'HalfCheetah_gated_3.txt', 'HalfCheetah_gated_4.txt', 'HalfCheetah_gated_5.txt']
Hopper = ['Hopper_gated_1.txt', 'Hopper_gated_2.txt', 'Hopper_gated_3.txt', 'Hopper_gated_4.txt', 'Hopper_gated_5.txt']
HumanoidStandup = ['HumanoidStandup_gated_1.txt', 'HumanoidStandup_gated_2.txt', 'HumanoidStandup_gated_3.txt', 'HumanoidStandup_gated_4.txt', 'HumanoidStandup_gated_5.txt']
Humanoid = ['Humanoid_gated_1.txt', 'Humanoid_gated_2.txt', 'Humanoid_gated_3.txt', 'Humanoid_gated_4.txt', 'Humanoid_gated_5.txt']
InvertedDoublePendulum = ['InvertedDoublePendulum_gated_1.txt', 'InvertedDoublePendulum_gated_2.txt', 'InvertedDoublePendulum_gated_3.txt', 'InvertedDoublePendulum_gated_4.txt', 'InvertedDoublePendulum_gated_5.txt']
InvertedPendulum = ['InvertedPendulum_gated_1.txt', 'InvertedPendulum_gated_2.txt', 'InvertedPendulum_gated_3.txt', 'InvertedPendulum_gated_4.txt', 'InvertedPendulum_gated_5.txt']
Pusher = ['Pusher_gated_1.txt', 'Pusher_gated_2.txt', 'Pusher_gated_3.txt', 'Pusher_gated_4.txt', 'Pusher_gated_5.txt']
Reacher = ['Reacher_gated_1.txt', 'Reacher_gated_2.txt', 'Reacher_gated_3.txt', 'Reacher_gated_4.txt', 'Reacher_gated_5.txt']
Striker = ['Striker_gated_1.txt', 'Striker_gated_2.txt', 'Striker_gated_3.txt', 'Striker_gated_4.txt', 'Striker_gated_5.txt']
Swimmer = ['Swimmer_gated_1.txt', 'Swimmer_gated_2.txt', 'Swimmer_gated_3.txt', 'Swimmer_gated_4.txt', 'Swimmer_gated_5.txt']
Thrower = ['Thrower_gated_1.txt', 'Thrower_gated_2.txt', 'Thrower_gated_3.txt', 'Thrower_gated_4.txt', 'Thrower_gated_5.txt']
Walker2d = ['Walker2d_gated_1.txt', 'Walker2d_gated_2.txt', 'Walker2d_gated_3.txt', 'Walker2d_gated_4.txt', 'Walker2d_gated_5.txt']





data = [Ant, HalfCheetah, Hopper, HumanoidStandup, Humanoid, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Striker, Swimmer, Thrower, Walker2d]

for i in range(len(data)):
    plt.subplot(4,4,i+1)
    plt.title(data[i][0].split('_')[0])
    plt.grid() 
    rewards = np.array(average(data[i]))
    rewards = np.cumsum(rewards)
    plt.plot(np.arange(len(rewards)), rewards)
plt.show()

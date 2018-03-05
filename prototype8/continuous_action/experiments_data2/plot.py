import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('..')
from experiments_data.plot import moving_average, get_rewards, average

   


'''
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
'''

Ant_ddpg = ['Ant_ddpg_1.txt', 'Ant_ddpg_2.txt', 'Ant_ddpg_3.txt']
Ant_gan = ['Ant_gan_3.txt', 'Ant_gan_4.txt', 'Ant_gated_1.txt']
Ant_gated = ['Ant_gated_2.txt', 'Ant_gated_5.txt']
HalfCheetah_ddpg = ['HalfCheetah_ddpg_1.txt', 'HalfCheetah_ddpg_2.txt', 'HalfCheetah_ddpg_3.txt']
HalfCheetah_gan = ['HalfCheetah_gan_3.txt', 'HalfCheetah_gan_4.txt', 'HalfCheetah_gan_5.txt']
HalfCheetah_gated = ['HalfCheetah_gated_1.txt', 'HalfCheetah_gated_2.txt', 'HalfCheetah_gated_3.txt', 'HalfCheetah_gated_5.txt']
Hopper_ddpg = ['Hopper_ddpg_1.txt', 'Hopper_ddpg_2.txt', 'Hopper_ddpg_3.txt']
Hopper_gan = ['Hopper_gan_3.txt', 'Hopper_gan_4.txt', 'Hopper_gan_5.txt']
Hopper_gated = ['Hopper_gated_1.txt', 'Hopper_gated_2.txt', 'Hopper_gated_3.txt', 'Hopper_gated_5.txt']
Humanoid_ddpg = ['Humanoid_ddpg_1.txt', 'Humanoid_ddpg_2.txt', 'Humanoid_ddpg_3.txt']
Humanoid_gan = ['Humanoid_gan_3.txt', 'Humanoid_gan_4.txt']
Humanoid_gated = ['Humanoid_gated_1.txt', 'Humanoid_gated_2.txt', 'Humanoid_gated_4.txt', 'Humanoid_gated_5.txt']
HumanoidStandup_ddpg = ['HumanoidStandup_ddpg_1.txt', 'HumanoidStandup_ddpg_2.txt', 'HumanoidStandup_ddpg_3.txt']
HumanoidStandup_gan = ['HumanoidStandup_gan_2.txt', 'HumanoidStandup_gan_3.txt', 'HumanoidStandup_gan_4.txt']
HumanoidStandup_gated = ['HumanoidStandup_gated_1.txt', 'HumanoidStandup_gated_2.txt', 'HumanoidStandup_gated_4.txt', 'HumanoidStandup_gated_5.txt']
InvertedDoublePendulum_ddpg = ['InvertedDoublePendulum_ddpg_2.txt', 'InvertedDoublePendulum_ddpg_3.txt']
InvertedDoublePendulum_gan = ['InvertedDoublePendulum_gan_3.txt', 'InvertedDoublePendulum_gan_4.txt', 'InvertedDoublePendulum_gan_5.txt']
InvertedDoublePendulum_gated = ['InvertedDoublePendulum_gated_1.txt', 'InvertedDoublePendulum_gated_2.txt', 'InvertedDoublePendulum_gated_3.txt', 'InvertedDoublePendulum_gated_5.txt']
InvertedPendulum_ddpg = ['InvertedPendulum_ddpg_2.txt', 'InvertedPendulum_ddpg_3.txt']
InvertedPendulum_gan = ['InvertedPendulum_gan_1.txt', 'InvertedPendulum_gan_4.txt', 'InvertedPendulum_gan_5.txt']
InvertedPendulum_gated = ['InvertedPendulum_gated_1.txt', 'InvertedPendulum_gated_2.txt', 'InvertedPendulum_gated_3.txt', 'InvertedPendulum_gated_5.txt']
Pusher_ddpg = ['Pusher_ddpg_2.txt', 'Pusher_ddpg_3.txt']
Pusher_gan = ['Pusher_gan_1.txt', 'Pusher_gan_4.txt', 'Pusher_gan_5.txt']
Pusher_gated = ['Pusher_gated_1.txt', 'Pusher_gated_2.txt', 'Pusher_gated_3.txt', 'Pusher_gated_4.txt']
Reacher_ddpg = ['Reacher_ddpg_2.txt', 'Reacher_ddpg_3.txt']
Reacher_gan = ['Reacher_gan_1.txt', 'Reacher_gan_2.txt', 'Reacher_gan_4.txt', 'Reacher_gan_5.txt']
Reacher_gated = ['Reacher_gated_1.txt', 'Reacher_gated_2.txt', 'Reacher_gated_3.txt', 'Reacher_gated_4.txt']
Striker_ddpg = ['Striker_ddpg_2.txt', 'Striker_ddpg_3.txt']
Striker_gan = ['Striker_gan_1.txt', 'Striker_gan_4.txt', 'Striker_gan_5.txt']
Striker_gated = ['Striker_gated_1.txt', 'Striker_gated_2.txt', 'Striker_gated_3.txt']
Swimmer_ddpg = ['Swimmer_ddpg_1.txt', 'Swimmer_ddpg_2.txt', 'Swimmer_ddpg_3.txt']
Swimmer_gan = ['Swimmer_gan_3.txt', 'Swimmer_gan_4.txt']
Swimmer_gated = ['Swimmer_gated_1.txt', 'Swimmer_gated_2.txt', 'Swimmer_gated_3.txt', 'Swimmer_gated_5.txt']
Thrower_ddpg = ['Thrower_ddpg_2.txt', 'Thrower_ddpg_3.txt']
Thrower_gan = ['Thrower_gan_1.txt', 'Thrower_gan_4.txt', 'Thrower_gan_5.txt']
Thrower_gated = ['Thrower_gated_1.txt', 'Thrower_gated_2.txt', 'Thrower_gated_3.txt']
Walker2d_ddpg = ['Walker2d_ddpg_1.txt', 'Walker2d_ddpg_2.txt', 'Walker2d_ddpg_3.txt']
Walker2d_gan = ['Walker2d_gan_3.txt', 'Walker2d_gan_4.txt']
Walker2d_gated = ['Walker2d_gated_1.txt', 'Walker2d_gated_2.txt', 'Walker2d_gated_3.txt', 'Walker2d_gated_5.txt']


data = [[Ant_ddpg, Ant_gan, Ant_gated], [HalfCheetah_ddpg, HalfCheetah_gan, HalfCheetah_gated], [Hopper_ddpg, Hopper_gan, Hopper_gated], [Humanoid_ddpg, Humanoid_gan, Humanoid_gated], [HumanoidStandup_ddpg, HumanoidStandup_gan, HumanoidStandup_gated], [InvertedDoublePendulum_ddpg, InvertedDoublePendulum_gan, InvertedDoublePendulum_gated], [InvertedPendulum_ddpg, InvertedPendulum_gan, InvertedPendulum_gated], [Pusher_ddpg, Pusher_gan, Pusher_gated], [Reacher_ddpg, Reacher_gan, Reacher_gated], [Striker_ddpg, Striker_gan, Striker_gated], [Swimmer_ddpg, Swimmer_gan, Swimmer_gated], [Thrower_ddpg, Thrower_gan, Thrower_gated], [Walker2d_ddpg, Walker2d_gan, Walker2d_gated]]


for i in range(len(data)):
    plt.subplot(4,4,i+1)
    plt.grid() 
    handles = []
    for j in range(len(data[i])):
        rewards = np.array(average(data[i][j], False))
        rewards = np.cumsum(rewards)
        #rewards, _ = moving_average(rewards, 50)
        handle, = plt.plot(np.arange(len(rewards)), rewards, label=data[i][j][0].split('.')[0][:-2])
        handles.append(handle)
    plt.title(data[i][0][0].split('_')[0])
    #plt.legend(handles=handles)
plt.show()

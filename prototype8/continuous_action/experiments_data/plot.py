import numpy as np
from matplotlib import pyplot as plt

def moving_average(data, N):
    avg = []
    std = []
    for i in range(len(data) - N + 1):
        avg.append(np.average(data[i:i+N]))
        std.append(np.std(data[i:i+N]))
    return np.asarray(avg), np.asarray(std)

def get_rewards(filepath):
    rewards = []
    with open(filepath) as fp:
        line = fp.readline()
        while line:
            if 'rewards' in line:
                l = line.strip().split(' ')
                rewards.append(float(l[-1]))
            line = fp.readline()
        return rewards

def main():
    Ant = ['Ant_ddpg_1.txt', 'Ant_gan_1.txt', 'Ant_gated_1.txt']
    HalfCheetah = ['HalfCheetah_ddpg_1.txt', 'HalfCheetah_gan_1.txt', 'HalfCheetah_gated_1.txt']
    Hopper = ['Hopper_ddpg_1.txt', 'Hopper_gan_1.txt', 'Hopper_gated_1.txt']
    HumanoidStandup = ['HumanoidStandup_ddpg_1.txt', 'HumanoidStandup_gan_1.txt', 'HumanoidStandup_gated_1.txt']
    Humanoid = ['Humanoid_ddpg_1.txt', 'Humanoid_gan_1.txt', 'Humanoid_gated_1.txt']
    InvertedDoublePendulum = ['InvertedDoublePendulum_ddpg_1.txt', 'InvertedDoublePendulum_gan_1.txt', 'InvertedDoublePendulum_gated_1.txt']
    InvertedPendulum = ['InvertedPendulum_ddpg_1.txt', 'InvertedPendulum_gan_1.txt', 'InvertedPendulum_gated_1.txt']
    Pusher = ['Pusher_ddpg_1.txt', 'Pusher_gan_1.txt', 'Pusher_gated_1.txt']
    Reacher = ['Reacher_ddpg_1.txt', 'Reacher_gan_1.txt', 'Reacher_gated_1.txt']
    Striker = ['Striker_ddpg_1.txt', 'Striker_gan_1.txt', 'Striker_gated_1.txt']
    Swimmer = ['Swimmer_ddpg_1.txt', 'Swimmer_gan_1.txt', 'Swimmer_gated_1.txt']
    Thrower = ['Thrower_ddpg_1.txt', 'Thrower_gan_1.txt', 'Thrower_gated_1.txt']
    Walker2d = ['Walker2d_ddpg_1.txt', 'Walker2d_gan_1.txt', 'Walker2d_gated_1.txt']

    data = [Ant, HalfCheetah, Hopper, HumanoidStandup, Humanoid, InvertedDoublePendulum, InvertedPendulum, Pusher, Reacher, Striker, Swimmer, Thrower, Walker2d]

    f = plt.figure(figsize=(12.0, 12.0))
    counter = 0
    for d in data:
        counter += 1
        plt.subplot(4,4,counter)
        handles = []
        for f in d:
            env = f.split('_')[0]
            model = f.split('_')[1]

            rewards = get_rewards(f)
            avg_rewards, std_rewards = moving_average(rewards, 100)

            plt.fill_between(np.arange(len(avg_rewards)), avg_rewards - std_rewards, avg_rewards + std_rewards, alpha=0.1)
            handle, = plt.plot(np.arange(len(avg_rewards)), np.array(avg_rewards), label=model)

            handles.append(handle)
        plt.legend(handles=handles)

        plt.grid()
        plt.xlabel('epoch')
        plt.ylabel('total rewards')
        plt.title(env)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()

python joint.py --environment=Reacher-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Reacher_gated_1.txt
python joint.py --environment=Pusher-v0 --model=gated --time-steps=10000 |tee ./experiments_data2/Pusher_gated_1.txt
python joint.py --environment=Thrower-v0 --model=gated --time-steps=10000 |tee ./experiments_data2/Thrower_gated_1.txt
python joint.py --environment=Striker-v0 --model=gated --time-steps=10000 |tee ./experiments_data2/Striker_gated_1.txt
python joint.py --environment=InvertedPendulum-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_gated_1.txt
python joint.py --environment=InvertedDoublePendulum-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_gated_1.txt
python joint.py --environment=HalfCheetah-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/HalfCheetah_gated_1.txt
python joint.py --environment=Hopper-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Hopper_gated_1.txt
python joint.py --environment=Swimmer-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Swimmer_gated_1.txt
python joint.py --environment=Walker2d-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Walker2d_gated_1.txt
python joint.py --environment=Ant-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Ant_gated_1.txt
python joint.py --environment=Humanoid-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Humanoid_gated_1.txt
python joint.py --environment=HumanoidStandup-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/HumanoidStandup_gated_1.txt
python joint.py --environment=Reacher-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Reacher_gan_1.txt
python joint.py --environment=Pusher-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Pusher_gan_1.txt
python joint.py --environment=Thrower-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Thrower_gan_1.txt
python joint.py --environment=Striker-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Striker_gan_1.txt
python joint.py --environment=InvertedPendulum-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_gan_1.txt
python joint.py --environment=InvertedDoublePendulum-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_gan_1.txt
python joint.py --environment=HalfCheetah-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/HalfCheetah_gan_1.txt
python joint.py --environment=Hopper-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Hopper_gan_1.txt
python joint.py --environment=Swimmer-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Swimmer_gan_1.txt
python joint.py --environment=Walker2d-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Walker2d_gan_1.txt
python joint.py --environment=Ant-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Ant_gan_1.txt
python joint.py --environment=Humanoid-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Humanoid_gan_1.txt
python joint.py --environment=HumanoidStandup-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/HumanoidStandup_gan_1.txt
python ddpg_refactored.py --environment=Reacher-v1 --time-steps=10000 |tee ./experiments_data2/Reacher_ddpg_1.txt
python ddpg_refactored.py --environment=Pusher-v0 --time-steps=10000 |tee ./experiments_data2/Pusher_ddpg_1.txt
python ddpg_refactored.py --environment=Thrower-v0 --time-steps=10000 |tee ./experiments_data2/Thrower_ddpg_1.txt
python ddpg_refactored.py --environment=Striker-v0 --time-steps=10000 |tee ./experiments_data2/Striker_ddpg_1.txt
python ddpg_refactored.py --environment=InvertedPendulum-v1 --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_ddpg_1.txt
python ddpg_refactored.py --environment=InvertedDoublePendulum-v1 --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_ddpg_1.txt

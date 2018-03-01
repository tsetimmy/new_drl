python joint.py --environment=InvertedPendulum-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_gated_5.txt
python joint.py --environment=InvertedDoublePendulum-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_gated_5.txt
python joint.py --environment=HalfCheetah-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/HalfCheetah_gated_5.txt
python joint.py --environment=Hopper-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Hopper_gated_5.txt
python joint.py --environment=Swimmer-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Swimmer_gated_5.txt
python joint.py --environment=Walker2d-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Walker2d_gated_5.txt
python joint.py --environment=Ant-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Ant_gated_5.txt
python joint.py --environment=Humanoid-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/Humanoid_gated_5.txt
python joint.py --environment=HumanoidStandup-v1 --model=gated --time-steps=10000 |tee ./experiments_data2/HumanoidStandup_gated_5.txt
python joint.py --environment=Reacher-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Reacher_gan_5.txt
python joint.py --environment=Pusher-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Pusher_gan_5.txt
python joint.py --environment=Thrower-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Thrower_gan_5.txt
python joint.py --environment=Striker-v0 --model=gan --time-steps=10000 |tee ./experiments_data2/Striker_gan_5.txt
python joint.py --environment=InvertedPendulum-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_gan_5.txt
python joint.py --environment=InvertedDoublePendulum-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_gan_5.txt
python joint.py --environment=HalfCheetah-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/HalfCheetah_gan_5.txt
python joint.py --environment=Hopper-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Hopper_gan_5.txt
python joint.py --environment=Swimmer-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Swimmer_gan_5.txt
python joint.py --environment=Walker2d-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Walker2d_gan_5.txt
python joint.py --environment=Ant-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Ant_gan_5.txt
python joint.py --environment=Humanoid-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/Humanoid_gan_5.txt
python joint.py --environment=HumanoidStandup-v1 --model=gan --time-steps=10000 |tee ./experiments_data2/HumanoidStandup_gan_5.txt
python ddpg_refactored.py --environment=Reacher-v1 --time-steps=10000 |tee ./experiments_data2/Reacher_ddpg_5.txt
python ddpg_refactored.py --environment=Pusher-v0 --time-steps=10000 |tee ./experiments_data2/Pusher_ddpg_5.txt
python ddpg_refactored.py --environment=Thrower-v0 --time-steps=10000 |tee ./experiments_data2/Thrower_ddpg_5.txt
python ddpg_refactored.py --environment=Striker-v0 --time-steps=10000 |tee ./experiments_data2/Striker_ddpg_5.txt
python ddpg_refactored.py --environment=InvertedPendulum-v1 --time-steps=10000 |tee ./experiments_data2/InvertedPendulum_ddpg_5.txt
python ddpg_refactored.py --environment=InvertedDoublePendulum-v1 --time-steps=10000 |tee ./experiments_data2/InvertedDoublePendulum_ddpg_5.txt
python ddpg_refactored.py --environment=HalfCheetah-v1 --time-steps=10000 |tee ./experiments_data2/HalfCheetah_ddpg_5.txt
python ddpg_refactored.py --environment=Hopper-v1 --time-steps=10000 |tee ./experiments_data2/Hopper_ddpg_5.txt
python ddpg_refactored.py --environment=Swimmer-v1 --time-steps=10000 |tee ./experiments_data2/Swimmer_ddpg_5.txt
python ddpg_refactored.py --environment=Walker2d-v1 --time-steps=10000 |tee ./experiments_data2/Walker2d_ddpg_5.txt
python ddpg_refactored.py --environment=Ant-v1 --time-steps=10000 |tee ./experiments_data2/Ant_ddpg_5.txt

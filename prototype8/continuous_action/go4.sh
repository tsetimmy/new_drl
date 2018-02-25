python ddpg.py --environment=Pusher-v0 --epochs=3000 |tee ./experiments_data/Pusher_ddpg_6.txt &&
python joint.py --environment=Pusher-v0 --model=gan --epochs=3000 |tee ./experiments_data/Pusher_gan_6.txt &&
python joint.py --environment=Pusher-v0 --model=gated --epochs=3000 |tee ./experiments_data/Pusher_gated_6.txt &&
python ddpg.py --environment=Thrower-v0 --epochs=3000 |tee ./experiments_data/Thrower_ddpg_6.txt &&
python joint.py --environment=Thrower-v0 --model=gan --epochs=3000 |tee ./experiments_data/Thrower_gan_6.txt &&
python joint.py --environment=Thrower-v0 --model=gated --epochs=3000 |tee ./experiments_data/Thrower_gated_6.txt &&
python ddpg.py --environment=Striker-v0 --epochs=3000 |tee ./experiments_data/Striker_ddpg_6.txt &&
python joint.py --environment=Striker-v0 --model=gan --epochs=3000 |tee ./experiments_data/Striker_gan_6.txt &&
python joint.py --environment=Striker-v0 --model=gated --epochs=3000 |tee ./experiments_data/Striker_gated_6.txt &&
python ddpg.py --environment=InvertedPendulum-v1 --epochs=3000 |tee ./experiments_data/InvertedPendulum_ddpg_6.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gan --epochs=3000 |tee ./experiments_data/InvertedPendulum_gan_6.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gated --epochs=3000 |tee ./experiments_data/InvertedPendulum_gated_6.txt &&
python ddpg.py --environment=Humanoid-v1 --epochs=3000 |tee ./experiments_data/Humanoid_ddpg_6.txt &&
python joint.py --environment=Humanoid-v1 --model=gan --epochs=3000 |tee ./experiments_data/Humanoid_gan_6.txt &&
python joint.py --environment=Humanoid-v1 --model=gated --epochs=3000 |tee ./experiments_data/Humanoid_gated_6.txt &&
python ddpg.py --environment=HumanoidStandup-v1 --epochs=3000 |tee ./experiments_data/HumanoidStandup_ddpg_6.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gan --epochs=3000 |tee ./experiments_data/HumanoidStandup_gan_6.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gated --epochs=3000 |tee ./experiments_data/HumanoidStandup_gated_6.txt

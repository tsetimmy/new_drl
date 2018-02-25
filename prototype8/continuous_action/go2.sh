python ddpg.py --environment=Pusher-v0 --epochs=3000 |tee ./experiments_data/Pusher_ddpg_2.txt &&
python joint.py --environment=Pusher-v0 --model=gan --epochs=3000 |tee ./experiments_data/Pusher_gan_2.txt &&
python joint.py --environment=Pusher-v0 --model=gated --epochs=3000 |tee ./experiments_data/Pusher_gated_2.txt &&
python ddpg.py --environment=Thrower-v0 --epochs=3000 |tee ./experiments_data/Thrower_ddpg_2.txt &&
python joint.py --environment=Thrower-v0 --model=gan --epochs=3000 |tee ./experiments_data/Thrower_gan_2.txt &&
python joint.py --environment=Thrower-v0 --model=gated --epochs=3000 |tee ./experiments_data/Thrower_gated_2.txt &&
python ddpg.py --environment=Striker-v0 --epochs=3000 |tee ./experiments_data/Striker_ddpg_2.txt &&
python joint.py --environment=Striker-v0 --model=gan --epochs=3000 |tee ./experiments_data/Striker_gan_2.txt &&
python joint.py --environment=Striker-v0 --model=gated --epochs=3000 |tee ./experiments_data/Striker_gated_2.txt &&
python ddpg.py --environment=InvertedPendulum-v1 --epochs=3000 |tee ./experiments_data/InvertedPendulum_ddpg_2.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gan --epochs=3000 |tee ./experiments_data/InvertedPendulum_gan_2.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gated --epochs=3000 |tee ./experiments_data/InvertedPendulum_gated_2.txt &&
python ddpg.py --environment=Humanoid-v1 --epochs=3000 |tee ./experiments_data/Humanoid_ddpg_2.txt &&
python joint.py --environment=Humanoid-v1 --model=gan --epochs=3000 |tee ./experiments_data/Humanoid_gan_2.txt &&
python joint.py --environment=Humanoid-v1 --model=gated --epochs=3000 |tee ./experiments_data/Humanoid_gated_2.txt &&
python ddpg.py --environment=HumanoidStandup-v1 --epochs=3000 |tee ./experiments_data/HumanoidStandup_ddpg_2.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gan --epochs=3000 |tee ./experiments_data/HumanoidStandup_gan_2.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gated --epochs=3000 |tee ./experiments_data/HumanoidStandup_gated_2.txt

python ddpg.py --environment=Pusher-v0 --epochs=3000 |tee ./experiments_data/Pusher_ddpg_3.txt &&
python joint.py --environment=Pusher-v0 --model=gan --epochs=3000 |tee ./experiments_data/Pusher_gan_3.txt &&
python joint.py --environment=Pusher-v0 --model=gated --epochs=3000 |tee ./experiments_data/Pusher_gated_3.txt &&
python ddpg.py --environment=Thrower-v0 --epochs=3000 |tee ./experiments_data/Thrower_ddpg_3.txt &&
python joint.py --environment=Thrower-v0 --model=gan --epochs=3000 |tee ./experiments_data/Thrower_gan_3.txt &&
python joint.py --environment=Thrower-v0 --model=gated --epochs=3000 |tee ./experiments_data/Thrower_gated_3.txt &&
python ddpg.py --environment=Striker-v0 --epochs=3000 |tee ./experiments_data/Striker_ddpg_3.txt &&
python joint.py --environment=Striker-v0 --model=gan --epochs=3000 |tee ./experiments_data/Striker_gan_3.txt &&
python joint.py --environment=Striker-v0 --model=gated --epochs=3000 |tee ./experiments_data/Striker_gated_3.txt &&
python ddpg.py --environment=InvertedPendulum-v1 --epochs=3000 |tee ./experiments_data/InvertedPendulum_ddpg_3.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gan --epochs=3000 |tee ./experiments_data/InvertedPendulum_gan_3.txt &&
python joint.py --environment=InvertedPendulum-v1 --model=gated --epochs=3000 |tee ./experiments_data/InvertedPendulum_gated_3.txt &&
python ddpg.py --environment=Humanoid-v1 --epochs=3000 |tee ./experiments_data/Humanoid_ddpg_3.txt &&
python joint.py --environment=Humanoid-v1 --model=gan --epochs=3000 |tee ./experiments_data/Humanoid_gan_3.txt &&
python joint.py --environment=Humanoid-v1 --model=gated --epochs=3000 |tee ./experiments_data/Humanoid_gated_3.txt &&
python ddpg.py --environment=HumanoidStandup-v1 --epochs=3000 |tee ./experiments_data/HumanoidStandup_ddpg_3.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gan --epochs=3000 |tee ./experiments_data/HumanoidStandup_gan_3.txt &&
python joint.py --environment=HumanoidStandup-v1 --model=gated --epochs=3000 |tee ./experiments_data/HumanoidStandup_gated_3.txt



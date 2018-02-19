python ddpg.py --environment=Reacher-v1 --epochs=1000 |tee ./experiments_data/Reacher_ddpg_1.txt &&
python ddpg.py --environment=Pusher-v0 --epochs=1000 |tee ./experiments_data/Pusher_ddpg_1.txt &&
python ddpg.py --environment=Thrower-v0 --epochs=1000 |tee ./experiments_data/Thrower_ddpg_1.txt &&
python ddpg.py --environment=Striker-v0 --epochs=1000 |tee ./experiments_data/Striker_ddpg_1.txt &&
python ddpg.py --environment=InvertedPendulum-v1 --epochs=1000 |tee ./experiments_data/InvertedPendulum_ddpg_1.txt &&
python ddpg.py --environment=InvertedDoublePendulum-v1 --epochs=1000 |tee ./experiments_data/InvertedDoublePendulum_ddpg_1.txt &&
python ddpg.py --environment=HalfCheetah-v1 --epochs=1000 |tee ./experiments_data/HalfCheetah_ddpg_1.txt &&
python ddpg.py --environment=Hopper-v1 --epochs=1000 |tee ./experiments_data/Hopper_ddpg_1.txt &&
python ddpg.py --environment=Swimmer-v1 --epochs=1000 |tee ./experiments_data/Swimmer_ddpg_1.txt &&
python ddpg.py --environment=Walker2d-v1 --epochs=1000 |tee ./experiments_data/Walker2d_ddpg_1.txt &&
python ddpg.py --environment=Ant-v1 --epochs=1000 |tee ./experiments_data/Ant_ddpg_1.txt &&
python ddpg.py --environment=Humanoid-v1 --epochs=1000 |tee ./experiments_data/Humanoid_ddpg_1.txt &&
python ddpg.py --environment=HumanoidStandup-v1 --epochs=1000 |tee ./experiments_data/HumanoidStandup_ddpg_1.txt

gnome-terminal -e "bash -c 'python joint.py --model=ddpg_unrolled_pg_mlp --time-steps=50000|tee ./ddpg_unrolledvanilla_pendulum1.txt'"
gnome-terminal -e "bash -c 'python joint.py --model=ddpg_unrolled_pg_mlp --time-steps=50000|tee ./ddpg_unrolledvanilla_pendulum2.txt'"
gnome-terminal -e "bash -c 'python joint.py --model=ddpg_unrolled_pg_mlp --time-steps=50000|tee ./ddpg_unrolledvanilla_pendulum3.txt'"
gnome-terminal -e "bash -c 'python joint.py --model=ddpg_unrolled_pg_mlp --time-steps=50000|tee ./ddpg_unrolledvanilla_pendulum4.txt'"
gnome-terminal -e "bash -c 'python joint.py --model=ddpg_unrolled_pg_mlp --time-steps=50000|tee ./ddpg_unrolledvanilla_pendulum5.txt'"

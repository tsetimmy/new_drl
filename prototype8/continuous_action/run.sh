python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=test |tee ./test_and_transfer/ddpg_test_1.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gan |tee ./test_and_transfer/jointgan_test_1.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gated |tee ./test_and_transfer/jointgated_test_1.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer |tee ./test_and_transfer/ddpg_transfer_1.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gan |tee ./test_and_transfer/jointgan_transfer_1.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gated |tee ./test_and_transfer/jointgated_transfer_1.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=test |tee ./test_and_transfer/ddpg_test_2.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gan |tee ./test_and_transfer/jointgan_test_2.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gated |tee ./test_and_transfer/jointgated_test_2.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer |tee ./test_and_transfer/ddpg_transfer_2.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gan |tee ./test_and_transfer/jointgan_transfer_2.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gated |tee ./test_and_transfer/jointgated_transfer_2.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=test |tee ./test_and_transfer/ddpg_test_3.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gan |tee ./test_and_transfer/jointgan_test_3.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gated |tee ./test_and_transfer/jointgated_test_3.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer |tee ./test_and_transfer/ddpg_transfer_3.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gan |tee ./test_and_transfer/jointgan_transfer_3.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gated |tee ./test_and_transfer/jointgated_transfer_3.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=test |tee ./test_and_transfer/ddpg_test_4.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gan |tee ./test_and_transfer/jointgan_test_4.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gated |tee ./test_and_transfer/jointgated_test_4.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer |tee ./test_and_transfer/ddpg_transfer_4.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gan |tee ./test_and_transfer/jointgan_transfer_4.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gated |tee ./test_and_transfer/jointgated_transfer_4.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=test |tee ./test_and_transfer/ddpg_test_5.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gan |tee ./test_and_transfer/jointgan_test_5.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=test --model=gated |tee ./test_and_transfer/jointgated_test_5.txt
python ddpg_refactored.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer |tee ./test_and_transfer/ddpg_transfer_5.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gan |tee ./test_and_transfer/jointgan_transfer_5.txt
python joint.py --time-steps=30000 --environment=Reacher-v1 --mode=transfer --model=gated |tee ./test_and_transfer/jointgated_transfer_5.txt

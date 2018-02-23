#!/bin/bash
#SBATCH --gres=gpu:1              # request GPU "generic resource"
#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=60G               # memory per node
#SBATCH --time=03-00:00            # time (DD-HH:MM)
#SBATCH --output=%N-%j.out        # %N for node name, %j for jobID

module load cuda cudnn python/2.7.13
source ~/tensorflow/bin/activate
python nature_replicate.py --environment=PongDeterministic-v4

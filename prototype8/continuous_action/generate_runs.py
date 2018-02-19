pre = '#!/bin/bash\n#SBATCH --gres=gpu:1              # request GPU "generic resource"\n#SBATCH --cpus-per-task=6    #Maximum of CPU cores per GPU request: 6 on Cedar, 16 on Graham.\n#SBATCH --mem=60G               # memory per node\n#SBATCH --time=03-00:00            # time (DD-HH:MM)\n#SBATCH --output=%N-%j.out        # %N for node name, %j for jobID\n\nmodule load cuda cudnn python/2.7.13\nsource ~/tensorflow/bin/activate\n'

iterations = 1
methods = ['ddpg.py', 'dyna.py', 'joint.py']
environments = ['HalfCheetah-v1', 'Swimmer-v1', 'Hopper-v1', 'Walker2d-v1', 'Ant-v1', 'Humanoid-v1']


for iteration in range(iterations):
    for method in methods:
        for env in environments:
            string = pre + 'python ' + method + ' ' + '--environment=' + env + ' ' + 'epochs=1000000000'

            text_file = open(method.split('.')[0] + '_' + env + '.sh', 'w')
            text_file.write(string)
            text_file.close()
            

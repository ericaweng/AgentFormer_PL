import subprocess
import torch

num_gpus = torch.cuda.device_count()
envs = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
env = 'univ'
# env = 'zara2'
cfgs = [f'{env}_sfm_pre_{i}' for i in ['.5',2,4,5,6,8,10,15]]
print("cfgs:", cfgs)
gpu_i = 0
for cfg in cfgs:
    # cmd = f"python test.py --cfg {cfg} --gpu {gpu_i} --epoch 10"
    cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --cached"
    print("cmd:", cmd)
    gpu_i += 1
    gpu_i %= num_gpus
    subprocess.Popen(cmd.split(' '))

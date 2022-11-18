"""parallelize saving all test trajectories"""

import os
import glob
import subprocess

# cfgs = os.listdir('results-o')
# cfgs = [c.split('/')[-1] for c in glob.glob('results/*weight-2.25_sigma_d-0.1')]
# for dataset in ['eth', 'univ', 'hotel', 'zara1', 'trajnet_sdd', 'zara2']:
#     for method in ['agentformer', 'pecnet', 'social_force', 'sgan', 'trajectron', 'vanilla_social_force', 'ynet', 'agentformer_sfm']:

cfgs = glob.glob('results-o/*')
for i, cfg in enumerate(cfgs):
    cfg_name = cfg.split('/')[-1]
    model_path = glob.glob(f'{cfg}/models/*p')
    # print("model_path:", model_path)
    model_path = model_path[0]
    env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": i,
    }
    cmd = f"python pl_train.py --cfg {cfg_name} -m test -lr results --save_traj -cp {model_path}"
    # cmd = f"python pl_train.py --cfg {cfg} -m test -lr results --save_traj"
    # cmd = f'python experiments/save_predicted_trajectories.py --sequence_name {dataset} --cfg {dataset}_agentformer_sfm_pre8-2{univ} --method agentformer_sfm'
    print("cmd:", cmd)
    # subprocess.Popen(cmd.split(' '))

# for cmd in scripts:
#     subprocess.Popen(cmd.split(' '))
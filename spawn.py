import os
import subprocess
# import multiprocessing
import torch
# multiprocessing.set_start_method('spawn')
# import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')


def main():
    num_gpus = torch.cuda.device_count()
    env = 'zara2'
    sigma_ds_weights = [(0.01, 0.05), (1.25, 0.0), (0.25, 1.5), (1, 0.01), (0.1, 1.5), (0.05, 0.01), (0.01, 0.0), (0.25, 0.1),
                        (0.5, 0.05), (0.75, 1.5), (0.5, 1.5), (0.025, 3), (0.25, 0.5), (0.5, 3), (0.01, 1.5), (1, 3),
                        (0.01, 2), (0.05, 3), (0.75, 0.1), (0.25, 0.0), (0.25, 3), (1, 0.0), (1.25, 0.01), (0.025, 0.5),
                        (1, 2), (0.1, 0.5), (0.5, 2), (0.025, 0.05), (0.01, 0.5), (1, 1), (0.5, 1), (0.05, 2), (0.1, 0.0),
                        (1.25, 0.05), (0.01, 3), (0.01, 0.1), (0.05, 0.05), (1, 1.5), (0.05, 0.1), (0.05, 1), (1.25, 1.5),
                        (1.25, 3), (0.75, 1), (0.025, 1.5), (0.25, 0.05), (1.25, 0.1), (0.75, 0.01), (0.1, 0.05), (0.05, 0.0),
                        (0.75, 0.0), (0.025, 0.01), (1.25, 0.5), (0.25, 0.01), (0.1, 1), (0.1, 2), (0.025, 1), (0.5, 0.1),
                        (1, 0.05), (0.75, 2), (0.01, 1), (0.75, 0.05), (0.05, 1.5), (0.1, 0.01), (0.25, 1), (0.025, 2),
                        (1.25, 2), (0.025, 0.0), (0.1, 3), (1, 0.1), (0.5, 0.5), (0.025, 0.1), (0.75, 3), (0.01, 0.01),
                        (0.75, 0.5), (0.5, 0.0), (1.25, 1), (0.05, 0.5), (0.1, 0.1), (0.5, 0.01), (0.25, 2), (1, 0.5)]

    cfg = f'{env}_sfm_base'#_weight-{weight}_sigma_d-{sigma_d}' for sigma_d, weight in sigma_ds_weights[:5]]

    gpu_i = 3
    cmds = []
    for sigma_d, weight in sigma_ds_weights[:2]:#for cfg in cfgs:
        cmd = f"python test.py --cfg {cfg} --gpu {gpu_i} --epoch last --weight {weight} --sigma_d {sigma_d} --cached"
        # cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --cached"
        print("cmd:", cmd)
        cmds.append(cmd)
        # gpu_i += 1
        # gpu_i %= num_gpus
        subprocess.Popen(cmd.split(' '), )

    # with mp.Pool() as pool:
    #     pool.map(os.system, tuple(cmds))


if __name__ == "__main__":
    main()

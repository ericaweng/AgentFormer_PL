import os
import subprocess
import glob
# import multiprocessing
import torch
# multiprocessing.set_start_method('spawn')
# import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')


def main(args):
    env = 'zara2'
    cfg = f'{env}_sfm_base'
    weights_sigma_ds = [(float(fp.split('_')[-3].split('-')[-1]), float(fp.split('_')[-1].split('-')[-1]))
                        for fp in glob.glob('./results/zara2_sfm_base_*')]
    if args.start_end is None:
        start, end = 0, len(weights_sigma_ds)
    else:
        start, end = args.start_end
    gpu_i = 0
    num_gpus = torch.cuda.device_count()
    # gpus = [4, 6]
    gpus = list(range(0, num_gpus))
    cmds = []
    for i, (weight, sigma_d) in enumerate(weights_sigma_ds[start:end]):#for cfg in cfgs:
        cmd = f"python test.py --cfg {cfg} --gpu {gpu_i} --epoch last --weight {weight} --sigma_d {sigma_d}"
        # cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --cached"
        print("cmd:", cmd)
        cmds.append(cmd)
        gpu_i = gpus[(i+1)%len(gpus)]
        # gpu_i += 1
        # gpu_i %= num_gpus
        subprocess.Popen(cmd.split(' '), )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_end', '-se', type=lambda x:map(int, x.split(',')))
    args = parser.parse_args()

    main(args)

"""test models nocol
and train cmd"""
import os
import time
import argparse
import torch
import subprocess


def main(args):
    cmds = []
    num_gpus = len(args.gpus_available)
    total_cmds = 0
    scene = 'zara2'
    for i in range(args.max_cmds):
        cmd = f"python pl_train.py --cfg zara2_agentformer_dlow_sfm --logs_root results-joint -ws"
        cmds.append(cmd)
        gpu_i = args.gpus_available[total_cmds % num_gpus]
        print(gpu_i, cmd)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu_i)}
        if not args.trial:
            subprocess.Popen(cmd.split(' '), env=env)
        time.sleep(1)
        total_cmds += 1
    print("total cmds:", total_cmds)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_cmds', '-mc', type=int, default=100)
    argparser.add_argument('--gpus_available', '-ga', nargs='+', type=int, default=list(range(torch.cuda.device_count())))
    argparser.add_argument('--trial', '-t', action='store_true')
    main(argparser.parse_args())

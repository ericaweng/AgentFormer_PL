import subprocess
import torch
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--gpu_start', '-gs', type=int, default=0)
parser.add_argument('--num', type=int, default=-1)
args = parser.parse_args()

num_gpus = torch.cuda.device_count()

sigma_ds = [0.25, 0.5, 0.75]#[ 0.01, 0.075, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 1.75, 2.]
# sigma_ds = [0.01, 0.25, 0.5, 0.75, 1, 1.25]#[ 0.01, 0.075, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 1.75, 2.]
# sigma_ds = [0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.]
weights = [1.5, 2.]#0., 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 3]#, 0.001, 0.005, 0.025, 0.075, 0.3, 1, 2, 3, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9]
# import itertools
# c = list(itertools.product(sigma_ds, weights))
# import random
# random.shuffle(c)
# print(c)

sigma_ds_weights = [(0.01, 0.05), (1.25, 0.0), (0.25, 1.5), (1, 0.01), (0.1, 1.5), (0.05, 0.01), (0.01, 0.0), (0.25, 0.1),
                    (0.5, 0.05), (0.75, 1.5), (0.5, 1.5), (0.025, 3), (0.25, 0.5), (0.5, 3), (0.01, 1.5), (1, 3),
                    (0.01, 2), (0.05, 3), (0.75, 0.1), (0.25, 0.0), (0.25, 3), (1, 0.0), (1.25, 0.01), (0.025, 0.5), (1, 2), (0.1, 0.5), (0.5, 2), (0.025, 0.05), (0.01, 0.5), (1, 1), (0.5, 1), (0.05, 2), (0.1, 0.0), (1.25, 0.05), (0.01, 3), (0.01, 0.1), (0.05, 0.05), (1, 1.5), (0.05, 0.1), (0.05, 1), (1.25, 1.5), (1.25, 3), (0.75, 1), (0.025, 1.5), (0.25, 0.05), (1.25, 0.1), (0.75, 0.01), (0.1, 0.05), (0.05, 0.0), (0.75, 0.0), (0.025, 0.01), (1.25, 0.5), (0.25, 0.01), (0.1, 1), (0.1, 2), (0.025, 1), (0.5, 0.1), (1, 0.05), (0.75, 2), (0.01, 1), (0.75, 0.05), (0.05, 1.5), (0.1, 0.01), (0.25, 1), (0.025, 2), (1.25, 2), (0.025, 0.0), (0.1, 3), (1, 0.1), (0.5, 0.5), (0.025, 0.1), (0.75, 3), (0.01, 0.01), (0.75, 0.5), (0.5, 0.0), (1.25, 1), (0.05, 0.5), (0.1, 0.1), (0.5, 0.01), (0.25, 2), (1, 0.5)]
# print("len(sigma_ds):", len(sigma_ds))
# print("len(weights):", len(weights))
# weights = [0., 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2]

# skip = 3
# start = 0
# sigma_ds_weights = []
sigma_ds_weights = itertools.product(sigma_ds, weights)
# for start in range(skip):
#     start += 1
#     for sigma_d in sigma_ds:
#         sigma_ds_weights.append((sigma_d, weights[start::skip]))


# for sigma_d in sigma_ds:
#     for weight in weights:
#         print(f"{sigma_d}\t{weight}")

gpu_i = args.gpu_start
total_running = -1
cfg = 'zara2_sfm_base'

if args.num == -1:
    args.num = len(sigma_ds) * len(weights) - args.start

for sigma_d, weight in sigma_ds_weights:
    total_running += 1
    if total_running < args.start:
        continue
    if total_running == args.start + args.num:
        break
    cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --sigma_d {sigma_d} --weight {weight}"
    print("cmd:", cmd)

    # print(f"{sigma_d}\t{weight}")
    gpu_i += 1
    gpu_i %= num_gpus
    subprocess.Popen(cmd.split(' '))

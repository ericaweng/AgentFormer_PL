import os
import subprocess
import glob
# import multiprocessing
import torch
# multiprocessing.set_start_method('spawn')
# import torch.multiprocessing as mp
# torch.multiprocessing.set_start_method('spawn')


def not_already_tested(fp):
    newness_thresh_epoch = 50
    globs = glob.glob(os.path.join(fp, 'results/*'))
    is_empty = len(globs) == 0
    last_epoch_low = False
    if not is_empty:
        last_epoch_low = int(sorted(globs)[-1][-4:]) < newness_thresh_epoch
        # last_epoch_not_complete = glob.glob(os.path.join(sorted(globs)[-1], ''))
    high_enough_model_exists = os.path.exists(os.path.join(fp, f'models/model_{newness_thresh_epoch:04d}.p'))
    return (is_empty or last_epoch_low) and high_enough_model_exists  # or last_epoch_not_complete


def not_already_tested(fp):
    return 'weight-2.0' in fp or 'weight-2.25' in fp or 'weight-2.0' in fp


def main(args):
    env = 'zara2'
    cfg = f'{env}_sfm_base'

    weights_sigma_ds = [(float(fp.split('_')[-3].split('-')[-1]), float(fp.split('_')[-1].split('-')[-1]))
                        for fp in glob.glob('./results/zara2_sfm_base_*')]  #  if not_already_tested(fp)
    print("weights_sigma_ds:", weights_sigma_ds)
    if args.start_end is None:
        start, end = 0, len(weights_sigma_ds)
    else:
        start, end = args.start_end
    gpu_start = 0
    num_gpus = torch.cuda.device_count()
    # gpus = [4, 6]
    gpus = list(range(gpu_start, num_gpus))
    cmds = []
    gpu_i = 0
    total_running = 0
    if args.max_running is None:
        args.max_running = num_gpus * 3
    for i, (weight, sigma_d) in enumerate(weights_sigma_ds[start:end]):#for cfg in cfgs:
        cmd = f"python test.py --cfg {cfg} --gpu {gpu_i} --all_epochs --weight {weight} --sigma_d {sigma_d} --resume"
        # cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --weight {weight} --sigma_d {sigma_d}"
        print("cmd:", cmd)
        cmds.append(cmd)
        gpu_i = gpus[(i+1)%len(gpus)]
        total_running += 1
        # gpu_i += 1
        # gpu_i %= num_gpus
        if total_running == args.max_running - 1:
            subprocess.run(cmd.split(' '))
            total_running = 0
        else:
            subprocess.Popen(cmd.split(' '))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_end', '-se', type=lambda x:map(int, x.split(',')))
    parser.add_argument('--max_running', '-mr', type=int, default=None)
    args = parser.parse_args()

    main(args)

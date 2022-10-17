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
    return 'weight-2.25' in fp and 'sigma_d-0.25' in fp or ('weight-2.5' in fp or 'weight-2.0' in fp) and ('sigma_d-1.0' in fp or 'sigma_d-1.25' in fp) or 'weight-2.75' in fp


def main(args):
    env = 'zara2'
    cfg = f'{env}_sfm_base'

    # weights_sigma_ds = [(float(fp.split('_')[-3].split('-')[-1]), float(fp.split('_')[-1].split('-')[-1]))
    #                     for fp in glob.glob('./results/zara2_sfm_base_*')]
    weights_sigma_ds = [(float(fp.split('_')[-3].split('-')[-1]), float(fp.split('_')[-1].split('-')[-1]))
                        for fp in glob.glob('./results/zara2_sfm_base_*') if not_already_tested(fp)]
    print("len(weights_sigma_ds):", len(weights_sigma_ds))

    if args.start_num is None:
        start, num = 0, len(weights_sigma_ds)
    else:
        start, num = args.start_num
    gpu_start = 0
    num_gpus = torch.cuda.device_count()
    # gpus = []
    gpus = list(range(gpu_start, num_gpus))
    cmds = []
    gpu_i = 0
    total_running = 0
    if args.max_running is None:
        args.max_running = num_gpus
    for i, (weight, sigma_d) in enumerate(weights_sigma_ds[start:start+num]):#for cfg in cfgs:
        # glob.glob(os.path.join(output_path, '*.pt'))
        # epochs = [int(p.split('_')[-1]) for p in glob.glob(f'results/zara2_sfm_base_weight-{weight}_sigma_d-{sigma_d}/results/epoch_*')]
        epochs = [int(p.split('_')[-1].split('.')[0]) for p in glob.glob(f'results/zara2_sfm_base_weight-{weight}_sigma_d-{sigma_d}/models/model_*.p') if p.split('_')[-1].split('.')[0] != 'last']
        # print("epochs:", epochs)
        epochs = ",".join([str(e) for e in epochs if e > 50])
        # print("epochs:", epochs)
        if len(epochs) == 0:
            continue
        cmd = f"python test.py --cfg {cfg} --gpu {gpu_i} --epochs {epochs} --weight {weight} --sigma_d {sigma_d}"
        # cmd = f"python train.py --cfg {cfg} --gpu {gpu_i} --weight {weight} --sigma_d {sigma_d}"
        print("cmd:", cmd)
        cmds.append(cmd)
        gpu_i = gpus[(i+1)%len(gpus)]
        total_running += 1
        # gpu_i += 1
        # gpu_i %= num_gpus
        # if total_running == args.max_running:
        #     subprocess.run(cmd.split(' '))
        #     total_running = 0
        # else:
        subprocess.Popen(cmd.split(' '))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_num', '-sn', type=lambda x:map(int, x.split(',')))
    parser.add_argument('--max_running', '-mr', type=int, default=None)
    args = parser.parse_args()

    main(args)

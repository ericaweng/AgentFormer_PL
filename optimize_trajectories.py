import glob
import os
import multiprocessing
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from .agentformer_loss import loss_func
from utils.utils import load_list_from_folder
from evaluate_all import peds_pandas_way, SEQUENCE_NAMES


def compute_traj_refinement_loss(self, data, total_loss):
    loss_funcs = {k: v for k, v in loss_func.items() if 'sfm' in k}
    for loss_name in loss_funcs:
        params = [data, self.loss_cfg.get(loss_name)]
        loss, _ = loss_funcs[loss_name](*params)
        total_loss = total_loss + loss
    return total_loss


def optimize_traj(self, data):
    data['train_dec_motion_old'] = data['train_dec_motion'].clone()
    data['infer_dec_motion_old'] = data['infer_dec_motion'].clone()
    data['train_dec_motion'] = nn.Parameter(data['train_dec_motion'])  # data_optim_train
    data['infer_dec_motion'] = nn.Parameter(data['infer_dec_motion'])  # data_optim_infer
    total_loss = Variable(torch.Tensor([0]), requires_grad=True)  #.to(data['pre_motion'].device)
    optimizer = optim.Adam([data['train_dec_motion'], data['infer_dec_motion']], lr=1e-4)  # self.cfg.lr)
    max_iters = 100
    for _ in range(max_iters):
        total_loss = self.compute_traj_refinement_loss(data, total_loss)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    return data


def main():
    save_dir = './trajectories_optimized'
    model_paths = glob.glob(save_dir)
    for model_path in model_paths:
        model = model_path.split('/')[-1]
    frames = []
    for seq in SEQUENCE_NAMES['zara1']:
        glob_str = f'{model_path}/{seq}/*'
        frames.extend(glob.glob(glob_str))

    # with multiprocessing.Pool(60) as pool:
    #     all_metrics = pool.map(do_one, frames)
    datas = []
    for frame_path in frames:
        sample_list, _ = load_list_from_folder(frame_path)
        sample_all = []
        gt = None
        if len(sample_list) == 0:
            print(f'No samples in {frame_path}')
            import ipdb; ipdb.set_trace()
        for sample in sample_list:
            if 'gt' in sample:
                gt = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
                gt = peds_pandas_way(gt, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id']).swapaxes(0, 1)
            if 'obs' in sample:
                obs = np.loadtxt(obs, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            if 'sample' not in sample:
                continue
            sample = np.loadtxt(sample, delimiter=' ', dtype='float32')  # (frames x agents) x 4
            sample = peds_pandas_way(sample, ['frame_id', 'ped_id', 'x', 'y'], ['frame_id', 'ped_id']).swapaxes(0, 1)
            sample_all.append(sample)

        samples = np.stack(sample_all, axis=0).swapaxes(0, 1)  # samples x (framex x agents) x 4
        assert gt is not None, os.listdir(frame_path)
        num_agents = gt.shape[0]
        assert num_agents == samples.shape[0]
        datas.append({'gt': gt, 'obs': obs})

    optimize_traj()


if __name__ == "__main__":
    main()
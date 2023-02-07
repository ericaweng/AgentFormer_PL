from functools import partial
import multiprocessing
import numpy as np
import torch

from model.model_lib import model_dict
from metrics import check_collision_per_sample_no_gt


def run_model_w_col_rej(data, model, traj_scale, sample_k, collision_rad, device):
    if isinstance(model, model_dict['dlow']):
        return per_sample_col_rej(data, model, traj_scale, sample_k, collision_rad, device)
    elif isinstance(model, model_dict['agentformer']):
        return col_rej(data, model, traj_scale, sample_k, collision_rad, device)

def per_sample_col_rej(data, model, traj_scale, sample_k, collision_rad, device):
    """run model with collision rejection per sample; for use with DLow w/ noise"""
    num_tries = 0
    num_zeros = 0
    MAX_NUM_SAMPLES = 300
    NUM_SAMPLES_PER_FORWARD = model.nk
    samples_to_return = None
    collision_info = None
    num_samples_wo_cols = model.nk
    sample_motion_3D_prev = None
    while samples_to_return.shape[0] < sample_k:
        with torch.no_grad():
            model.set_data(data)
            sample_motion_3D = model.inference(mode='infer', sample_num=NUM_SAMPLES_PER_FORWARD,
                                               need_weights=False)[0].transpose(0, 1).contiguous()
            if sample_motion_3D_prev is not None:
                assert torch.any(sample_motion_3D_prev != sample_motion_3D)
            sample_motion_3D_prev = sample_motion_3D
            num_tries += 1
        sample_motion_3D *= traj_scale

        # compute number of colliding samples
        pred_arr = sample_motion_3D.cpu().numpy()
        num_peds = pred_arr.shape[1]
        if num_peds == 1:  # if there's only one ped, there are necessarily no collisions
            samples_to_return = sample_motion_3D[:sample_k]
            break
        # compute collisions in parallel
        with multiprocessing.Pool(processes=min(NUM_SAMPLES_PER_FORWARD, multiprocessing.cpu_count())) as pool:
            mask = pool.map(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
        # get indices of samples that have 0 collisions
        maskk = np.where(~np.any(np.array(list(zip(*mask))[0]).astype(np.bool), axis=-1))[0]
        # num_samples_wo_cols = maskk.shape[0]
        if num_samples_wo_cols == 0:  # if there are no samples with 0 collisions
            num_zeros += 1
            if num_tries * NUM_SAMPLES_PER_FORWARD >= MAX_NUM_SAMPLES:
                # if num_zeros > MAX_NUM_ZEROS or num_tries > MAX_NUM_TRIES:
                print(f"frame {data['frame']} with {len(data['pre_motion_3D'])} peds: "
                  f"collected {num_tries * NUM_SAMPLES_PER_FORWARD} samples, only {num_samples_wo_col} non-colliding. \n")
                collision_info = num_peds, model.nk - num_samples_wo_cols
                break
            continue
        non_collide_idx = torch.LongTensor(maskk)
        assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
        assert 0 <= torch.max(non_collide_idx)
        samples_to_return = torch.index_select(sample_motion_3D, 0, non_collide_idx.to(device))  # select only those in current sample who don't collide

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    samples_to_return = samples_to_return.transpose(0, 1)
    return samples_to_return.cpu(), gt_motion_3D.cpu(), collision_info


def col_rej(data, model, traj_scale, sample_k, collision_rad, device):
    """run model with collision rejection"""
    num_tries = 0
    num_zeros = 0
    MAX_NUM_SAMPLES = 300
    NUM_SAMPLES_PER_FORWARD = 30
    samples_w_cols = None
    sample_motion_3D_prev = None
    samples_to_return = None
    while samples_to_return.shape[0] < sample_k:
        with torch.no_grad():
            model.set_data(data)
            sample_motion_3D = model.inference(mode='infer', sample_num=NUM_SAMPLES_PER_FORWARD,
                                               need_weights=False)[0].transpose(0, 1).contiguous()
            if sample_motion_3D_prev is not None:
                assert torch.any(sample_motion_3D_prev != sample_motion_3D), "shouldn't use col rej with dlow w/o noise"
            sample_motion_3D_prev = sample_motion_3D
            if samples_to_return is None:
                samples_to_return = sample_motion_3D
            num_tries += 1
        sample_motion_3D *= traj_scale

        # compute number of colliding samples
        pred_arr = sample_motion_3D.cpu().numpy()
        num_peds = pred_arr.shape[1]
        if num_peds == 1:  # if there's only one ped, there are necessarily no collisions
            break
        # compute collisions in parallel
        with multiprocessing.Pool(processes=min(NUM_SAMPLES_PER_FORWARD, multiprocessing.cpu_count())) as pool:
            mask = pool.map(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
            # no_mp alternative:
            # mask = itertools.starmap(partial(check_collision_per_sample_no_gt, ped_radius=collision_rad), pred_arr)
            # mask contains list of length num_samples of tuples of length 2
            # (collision_per_ped_array (num_peds), collision_matrix_per_timestep (pred_steps, num_peds, num_peds))
        # get indices of samples that have 0 collisions
        maskk = np.where(~np.any(np.array(list(zip(*mask))[0]).astype(np.bool), axis=-1))[0]
        if maskk.shape[0] == 0:  # if there are no samples with 0 collisions
            num_zeros += 1
            if num_tries * NUM_SAMPLES_PER_FORWARD >= MAX_NUM_SAMPLES:
                # if num_zeros > MAX_NUM_ZEROS or num_tries > MAX_NUM_TRIES:
                print(f"frame {data['frame']} with {len(data['pre_motion_3D'])} peds: "
                  f"collected {num_tries * NUM_SAMPLES_PER_FORWARD} samples, only {non_collide_idx} non-colliding. \n")
                samples_w_cols = num_peds, sample_k - samples_to_return.shape[0]
                break
            continue
        # append new non-colliding samples to list
        # at_least_1_col = np.any([np.any([np.any(ped) for ped in sample]) for sample in mask])
        # if at_least_1_col:
        #     print(f"Seq {data['seq']} frame {data['frame']} with {len(data['pre_motion_3D'])} peds has {maskk.shape[0]} non-colliding samples in 50 samples")
        non_collide_idx = torch.LongTensor(maskk)
        assert torch.max(non_collide_idx) < sample_motion_3D.shape[0]
        assert 0 <= torch.max(non_collide_idx)
        samples_to_return[non_collide_idx] = sample_motion_3D[non_collide_idx]

    gt_motion_3D = torch.stack(data['fut_motion_3D'], dim=0).to(device) * traj_scale
    samples_to_return = samples_to_return.transpose(0, 1)
    return samples_to_return.cpu(), gt_motion_3D.cpu(), samples_w_cols

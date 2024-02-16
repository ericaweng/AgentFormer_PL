"""
Contains code for loading, preprocessing
and visualizing parts of the H3.6M dataset.

- need to add normalization
- need to make sure the agentformer actually works for 3d
- support all the random augmentations
- need to move "set_data" into the dataloader.
"""
import copy
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append(os.getcwd())

from utils.utils import print_log


class PedXDataset(Dataset):
    def __init__(self, cfg, log, split='train', phase='training'):
        self.cfg = cfg
        self.past_frames = cfg.past_frames
        self.min_past_frames = cfg.min_past_frames
        self.frame_skip = cfg.get('frame_skip', 1)
        self.phase = phase
        self.split = split

        assert cfg.dataset == "pedx"

        data_root = cfg.data_root_pedx
        self.data_root = data_root
        self.data_file = os.path.join(data_root, 'pedx_joint_pos.npz')
        self.splits_file = os.path.join(data_root, 'pedx_joint_pos_splits.npz')
        # FOr now, only train on a small subsequence.
        # remove this later since we don't diff the seqs, just the subj

        # will need to sep it by subject later when supporting multiple things.
        self.sequence_to_load = []

        self.past_frames = cfg.past_frames
        self.future_frames = cfg.future_frames
        self.frame_skip = cfg.get('frame_skip', 1)
        self.min_past_frames = cfg.get('min_past_frames', self.past_frames)
        self.min_future_frames = cfg.get('min_future_frames', self.future_frames)
        self.traj_scale = cfg.traj_scale
        self.past_traj_scale = cfg.traj_scale
        # h36m kp mask
        # self.kp_mask = np.array([1,2,3,4,5,6,7,8,9,13,14,15,16,18,19,20,26,27,28])-1  # 19 kp
        # pedx kp mask
        self.kp_mask = np.arange(24)  # ([0,1,2,3,4,5,6,7,8,9,10,11,12,16,18,19,20,26,27,28])-1  # 19 kp
        self.num_kp = len(self.kp_mask)
        self.split = split
        self.phase = phase
        self.log = log

        print_log("\n-------------------------- loading %s data --------------------------" % split, log=log)
        self.num_total_samples = 0
        self.num_sample_list = []
        self.all_kp_data = np.load(self.data_file, allow_pickle=True)['kp'].item()
        self.all_trajs_data = np.load(self.data_file, allow_pickle=True)['pos'].item()
        self.subjects_split = np.load(self.splits_file, allow_pickle=True)['data'].item()
        # for each capture date, change all_data to only contain the frames in each split
        self.num_frames_total = 0
        for capture_date in self.subjects_split[split]:
            print(f"split: {split}")
            min_frame_id = min(self.subjects_split[split][capture_date])
            max_frame_id = max(self.subjects_split[split][capture_date])
            min_frame_idx = list(self.all_kp_data[capture_date].keys()).index(min_frame_id)
            print(f"min_frame_idx: {min_frame_idx}")
            max_frame_idx = list(self.all_kp_data[capture_date].keys()).index(max_frame_id)
            print(f"max_frame_idx: {max_frame_idx}")
            # verify
            min_frame_idx_pos = list(self.all_trajs_data[capture_date].keys()).index(min_frame_id)
            assert min_frame_idx_pos == min_frame_idx, f"min_frame_idx_pos ({min_frame_idx_pos}) must be equal to min_frame_idx ({min_frame_idx})"
            max_frame_idx_pos = list(self.all_trajs_data[capture_date].keys()).index(max_frame_id)
            assert max_frame_idx_pos == max_frame_idx, f"max_frame_idx_pos ({max_frame_idx_pos}) must be equal to max_frame_idx ({max_frame_idx})"

            # other verify
            assert min_frame_idx < max_frame_idx, f"min_frame_idx ({min_frame_idx}) must be less than max_frame_idx ({max_frame_idx})"
            assert max_frame_idx < len(self.all_kp_data[
                                           capture_date].keys()), f"max_frame_idx ({max_frame_idx}) must be less than len(self.all_data[capture_date].keys()) ({len(self.all_kp_data[capture_date].keys())})"

            print("self.all_data[capture_date].shape:", list(self.all_kp_data[capture_date].values())[0]
            # self.all_kp_data[capture_date] = np.array(sorted(self.all_kp_data[capture_date].items(), key=lambda x: x[0]))[
            #                     min_frame_idx:max_frame_idx + 1]
            self.all_kp_data[capture_date] = {k:v for k,v in self.all_kp_data[capture_date].items() if k >= min_frame_id and k <= max_frame_id}

            # check that frame_ids are equally-spaced
            frame_ids = sorted(map(int, self.all_kp_data[capture_date].keys()))
            frame_ids_diff = np.diff(frame_ids)
            assert np.all(frame_ids_diff == frame_ids_diff[0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

            print("self.all_data[capture_date].shape:", np.array(self.all_kp_data[capture_date].values()).shape)
            self.num_frames_total += len(self.all_kp_data[capture_date])
        print(f"num_frames_total: {self.num_frames_total}")
        import ipdb; ipdb.set_trace()

        self.seq_labels = defaultdict(list)
        for capture_date in self.subjects_split[split]:
            print_log(f"loading capture date {capture_date} ...", log=log)
            seq_exists = self.load_sequence(capture_date)
            if seq_exists:
                seq_name = capture_date
                self.sequence_to_load.append(seq_name)

        self.index = 0

        print_log(f'total num samples: {self.num_total_samples}', log)
        print_log("------------------------------ done --------------------------------\n", log=log)

    def load_sequence(self, capture_date):
        # make the input data dictionary into a numpy array
        start_frame = 0
        num_seq_samples = self.all_kp_data.shape[0] - (self.min_past_frames + self.min_future_frames - 1) * self.frame_skip
        end_frame = start_frame + num_seq_samples
        num_valid_samples = 0
        for frame_idx in range(start_frame, end_frame):
            data = self.get_sample(frame_idx, capture_date)
            if data is not None:
                self.seq_labels[f'{capture_date}'].append(data)
                num_valid_samples += 1

        self.num_total_samples += num_valid_samples
        self.num_sample_list.append(num_valid_samples)

        return num_valid_samples > 0

    def get_pre_data(self, frame_idx, capture_date):
        joint_data = self.all_kp_data[capture_date]
        traj_data = self.all_pos_data[capture_date]
        history_kp = np.zeros((self.past_frames, self.num_kp, 3))
        for i in range(self.past_frames):
            data = joint_data[frame_idx - i * self.frame_skip][self.kp_mask]
            history_kp[i] = data
        return torch.tensor(history_kp)

    def get_future_data(self, frame_idx, subject_idx, action):
        subject_data = self.all_kp_data[subject_idx][action]
        future = np.zeros((self.future_frames, self.num_kp, 3))
        for i in range(1, self.future_frames + 1):
            data = subject_data[frame_idx + i * self.frame_skip][self.kp_mask]
            future[i - 1] = data
        return torch.tensor(future)

    def get_sample(self, frame_idx, capture_date):
        """
        We assume H3.6M has valid IDs at every timestep.
        Every joint occurs at every single timestamp as well.
        """
        pre_data = self.get_pre_data(frame_idx, capture_date)
        future_data = self.get_future_data(frame_idx, capture_date)
        num_agents = pre_data.shape[1]
        if len(pre_data[0]) == 0 or len(future_data[0]) == 0:
            return None

        pre_motion_3d = pre_data
        pre_motion_mask = torch.ones((pre_data.shape[0], pre_data.shape[1]))
        # num_agents (all_valid) * num_fut_frames * 3
        fut_motion_3d = future_data
        fut_motion_mask = torch.ones((future_data.shape[0], future_data.shape[1]))
        # what's the point of this traj_scale paramater.
        traj_scale = 1

        data = {
                'pre_motion': pre_motion_3d.to(torch.float32),
                'fut_motion': fut_motion_3d.to(torch.float32),
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': pre_data.to(torch.float32),
                'fut_data': future_data.to(torch.float32),
                'valid_id': torch.arange(num_agents),
                'traj_scale': traj_scale,
                'seq': f"{capture_date}_{action}",
                'frame': frame_idx
        }

        return data

    def get_seq_and_frame(self, index):
        index_tmp = copy.copy(index)
        for seq_index in range(len(self.num_sample_list)):  # 0-indexed
            if index_tmp < self.num_sample_list[seq_index]:
                # from 0-indexed list index to 1-indexed frame index (for mot)
                # I think we don't need this, since we've removed a lot of the nones.
                # this would just pad nones..>
                frame_index = index_tmp  # + (self.min_past_frames - 1) * self.frame_skip + self.sequence[seq_index].init_frame
                return seq_index, frame_index
            else:
                index_tmp -= self.num_sample_list[seq_index]

        assert False, 'index is %d, out of range' % (index)

    def __len__(self):
        return self.num_total_samples

    def __getitem__(self, idx):
        seq_index, frame = self.get_seq_and_frame(idx)
        seq_name = self.sequence_to_load[seq_index]

        in_data = self.seq_labels[seq_name][frame]
        data = defaultdict(lambda: None)
        data['batch_size'] = self.cfg.batch_size
        data['agent_num'] = in_data['pre_motion'].shape[1]
        data['pre_motion'] = in_data['pre_motion'].contiguous()
        data['fut_motion'] = in_data['fut_motion'].contiguous()
        data['fut_motion_orig'] = in_data['fut_motion'].transpose(0, 1).contiguous()
        data['pre_motion_orig'] = in_data['pre_motion'].transpose(0, 1).contiguous()
        data['fut_mask'] = in_data['fut_motion_mask']
        data['pre_mask'] = in_data['pre_motion_mask']

        data['seq'] = seq_index
        data['frame'] = frame
        scene_orig_all_past = self.cfg.get('scene_orig_all_past', False)
        if scene_orig_all_past:
            data['scene_orig'] = data['pre_motion'].view(-1, 3).mean(dim=0)
        else:
            data['scene_orig'] = data['pre_motion'][-1].mean(dim=0)

        # theta = torch.zeros(1).
        for key in ['pre_motion', 'fut_motion', 'fut_motion_orig']:
            data[f'{key}_scene_norm'] = data[key] - data['scene_orig']  # normalize per scene

        data['pre_vel'] = data['pre_motion'][1:] - data['pre_motion'][:-1, :]
        data['fut_vel'] = data['fut_motion'] - torch.cat([data['pre_motion'][[-1]], data['fut_motion'][:-1, :]])
        data['cur_motion'] = data['pre_motion'][[-1]]
        data['pre_motion_norm'] = data['pre_motion'][:-1] - data['cur_motion']  # normalize pos per agent
        data['fut_motion_norm'] = data['fut_motion'] - data['cur_motion']

        conn_dist = self.cfg.get('conn_dist', 100000.0)
        cur_motion = data['cur_motion'][0]
        if conn_dist < 1000.0:
            threshold = conn_dist / self.cfg.traj_scale
            pdist = F.pdist(cur_motion)
            D = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]])
            D[np.triu_indices(cur_motion.shape[0], 1)] = pdist
            D += D.T
            mask = torch.zeros_like(D)
            mask[D > threshold] = float('-inf')
        else:
            mask = torch.zeros([cur_motion.shape[0], cur_motion.shape[0]])
        data['agent_mask'] = mask

        return data


import torch
import numpy as np
import pandas as pd
from pathlib import Path


class TBDPreprocess:
    def __init__(self, data_root, seq_name, parser, split='train', phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.include_cats = parser.get('include_cats', [])
        self.exclude_kpless_data = parser.get('exclude_kpless_data', False)
        self.zero_kpless_data = parser.get('zero_kpless_data', False)
        self.data_root = data_root
        self.past_frames = parser.past_frames
        self.past_frames_pos = parser.get('past_frames_pos', self.past_frames)
        self.past_frames_kp = parser.get('past_frames_kp', self.past_frames)
        self.future_frames = parser.future_frames
        self.frame_skip = parser.get('frame_skip', 1)
        self.min_past_frames = parser.get('min_past_frames', self.past_frames)
        self.min_future_frames = parser.get('min_future_frames', self.future_frames)
        self.traj_scale = parser.traj_scale
        self.past_traj_scale = parser.traj_scale
        self.load_map = parser.get('load_map', False)
        self.map_version = parser.get('map_version', '0.1')
        self.seq_name = seq_name
        self.split = split
        self.phase = phase

        self.gt = np.genfromtxt(f'{data_root}/{seq_name}.txt', delimiter=' ', dtype=float)
        if np.any(['kp' in i for i in parser.input_type]) or np.any(['ori' in i for i in parser.input_type]):
            self.kp_source = parser.get('kp_source', 'hmr2')
            if self.kp_source == 'blazepose':
                # self.all_kp_data = np.load(f'{data_root}/agent_keypoints/{seq_name}_kp.npz', allow_pickle=True)['arr_0'].item()
                # self.kp_mask = np.arange(33)
                import ipdb; ipdb.set_trace()
            elif self.kp_source == 'hmr2':
                self.all_kp_data = np.load(f"{Path(data_root).parent.parent.parent}/tbd_hmr2_raw/{seq_name}_kp3d.npz", allow_pickle=True)['arr_0'].item()
                self.kp_mask = np.arange(44)
            else:
                raise ValueError(f"kp_source {self.kp_source} not recognized")
        else:
            self.kp_mask = np.arange(44)
        if parser.get('include_robot_data', False):
            self.robot_data = pd.read_csv(f'{data_root}/robot_poses/{seq_name}_robot.txt', delimiter=' ', dtype=float, names=['x', 'y', 'yaw']).rename_axis('timestep')
            pos_2d = self.robot_data[['x', 'y']].values
            self.robot_data['p'] = list(np.concatenate([pos_2d, np.zeros((pos_2d.shape[0], 1))], axis=1))
            self.robot_data = self.robot_data[['p', 'yaw']]
        else:
            self.robot_data = None

        frames = np.unique(self.gt[:, 0].astype(int))
        # frame_ids_diff = np.diff(frames)
        # assert np.all(frame_ids_diff == frame_ids_diff[0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

        fr_start, fr_end = frames.min(), frames.max()
        # subtract fr_start to make frame ids start from 0
        self.gt[:, 0] -= fr_start
        fr_end -= fr_start
        fr_start = 0
        if 'half_and_half' in parser.get('split_type', 'normal'):
            train_size = len(frames) * 3 // 4
            test_size = len(frames) * 1 // 4
            if split == 'train':
                self.num_fr = train_size
                self.init_frame = fr_start
            else:
                self.num_fr = test_size
                self.init_frame = fr_start + train_size
        else:
            self.num_fr = len(frames)
            self.init_frame = fr_start

        self.geom_scene_map = None
        self.gt = self.gt.astype('float32')
        self.xind, self.zind = 2, 3
        self.heading_ind = 4

    def get_id(self, data):
        return data[:, 1].copy()

    def total_frames(self):
        return self.num_fr

    def get_data(self, frame_id):
        if hasattr(self, 'all_kp_data') and frame_id in self.all_kp_data:
            data_kp = self.all_kp_data[frame_id]
        else:
            data_kp = {}

        data_pos = self.gt[self.gt[:, 0] == frame_id]
        return {'kp': data_kp, 'pos': data_pos}

    def get_pre_data(self, frame):
        data_list = []
        for i in range(self.past_frames):
            frame_id = frame - i * self.frame_skip
            data_list.append(self.get_data(frame_id))
        return data_list

    def get_fut_data(self, frame):
        data_list = []
        for i in range(1, self.future_frames + 1):
            frame_id = frame + i * self.frame_skip
            data_list.append(self.get_data(frame_id))
        return data_list

    def get_valid_id_pos_and_kp(self, pre_data, fut_data):
        valid_id = []
        for idx in np.unique(self.get_id(pre_data[0]['pos'])):
            is_invalid = False
            for frame in pre_data[:self.min_past_frames]:
                exclude_kp = self.exclude_kpless_data and (
                    idx not in frame['kp']
                    or frame['kp'][idx] is None
                    or np.all(frame['kp'][idx] == 0)
                    or np.all(np.isnan(frame['kp'][idx]))
                )
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:, 1] or exclude_kp:
                    is_invalid = True
                    break
            if is_invalid:
                continue
            for frame_i, frame in enumerate(fut_data[:self.min_future_frames]):
                exclude_kp = self.exclude_kpless_data and (
                    idx not in frame['kp']
                    or frame['kp'][idx] is None
                    or np.all(frame['kp'][idx] == 0)
                    or np.all(np.isnan(frame['kp'][idx]))
                )
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:, 1] or exclude_kp:
                    is_invalid = True
                    break
            if is_invalid:
                continue
            if len(self.include_cats) > 0:
                history_positions = np.array([p['pos'][p['pos'][:, 1] == idx][0][2:4] for p in pre_data])
                future_positions = np.array([p['pos'][p['pos'][:, 1] == idx][0][2:4] for p in fut_data])
                pos = np.concatenate([history_positions, future_positions], axis=0)
                if ped_interactions.is_moving_to_static(pos)[0] \
                    or ped_interactions.is_static_to_moving(pos)[0] \
                    or ped_interactions.is_non_linear(pos)[0]:
                    continue
            valid_id.append(idx)

        return valid_id

    def get_heading(self, cur_data, valid_id):
        headings = []
        for i, idx in enumerate(valid_id):
            headings.append(cur_data['kp'][cur_data['kp'][:, 1] == idx].squeeze()[self.heading_ind])
        assert len(headings) == len(valid_id)
        return headings

    def get_heading_avg(self, all_data, valid_id):
        avg_headings = []
        for i, idx in enumerate(valid_id):
            headings_this_ped = []
            for ts in range(len(all_data)):
                h = all_data[ts]['pos'][all_data[ts]['pos'][:, 1] == idx].squeeze()
                if len(h) == 0:
                    continue
                h = h[self.heading_ind]
                headings_this_ped.append((np.cos(h), np.sin(h)))
            heading_avg = np.stack(headings_this_ped).mean(0)
            avg_headings.append(np.arctan2(heading_avg[1], heading_avg[0]))
        return avg_headings

    def format_data(self, data, num_frames, valid_id, is_pre=False):
        motion = []
        mask = []
        kp_motion = []
        for ped_id in valid_id:
            mask_i = torch.zeros(num_frames)
            box_3d = torch.zeros([num_frames, 2])
            kp_3d = torch.zeros([num_frames, len(self.kp_mask), 3])
            for f_i, frame_i in enumerate(range(num_frames)):
                single_frame = data[frame_i]
                pos_history = single_frame['pos']

                if is_pre:
                    frame_i = num_frames - 1 - frame_i
                if not self.zero_kpless_data or (self.zero_kpless_data and single_frame['kp'] is not None and ped_id in single_frame['kp']):
                    found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[[self.xind, self.zind]] / self.past_traj_scale
                    assert len(found_data) != 0
                    box_3d[frame_i] = torch.from_numpy(found_data).float()
                    mask_i[frame_i] = 1.0
                else:
                    box_3d[frame_i] = torch.zeros((1, 2))
                    mask_i[frame_i] = 0
                if single_frame['kp'] is not None and ped_id in single_frame['kp'] and single_frame['kp'][ped_id] is not None:
                    kp_3d[frame_i] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                else:
                    kp_3d[frame_i] = 0
            motion.append(box_3d)
            mask.append(mask_i)
            kp_3d[kp_3d != kp_3d] = 0
            kp_motion.append(kp_3d)

        return motion, kp_motion, mask

    def get_formatted_pre_data(self, data, valid_id):
        return self.format_data(data, self.past_frames, valid_id, is_pre=True)

    def get_formatted_fut_data(self, data_future, valid_id):
        return self.format_data(data_future, self.future_frames, valid_id)

    def get_interaction_mask(self, pre_data, fut_data, valid_id):
        mask = np.zeros(len(valid_id))
        for idx, ped_idx in valid_id:
            history_positions = np.array([p['pos'][p['pos'][:, 1] == ped_idx][0][2:4] for p in pre_data])
            future_positions = np.array([p['pos'][p['pos'][:, 1] == ped_idx][0][2:4] for p in fut_data])
            pos = np.concatenate([history_positions, future_positions], axis=0)
            if ped_interactions.is_static(pos)[0]:
                mask[idx] = 1
        return mask

    def get_relevant_robot_data(self, frame):
        start_frame = frame - self.past_frames * self.frame_skip
        end_frame = frame + (1 + self.future_frames) * self.frame_skip
        relevant_frames = range(start_frame, end_frame, self.frame_skip)
        curr_robot_data = self.robot_data[self.robot_data.index.get_level_values('timestep').astype(int).isin(relevant_frames)]
        return curr_robot_data

    def __call__(self, frame):
        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.total_frames() - 1, 'frame is %d, total is %d' % (frame, self.total_frames())

        pre_data = self.get_pre_data(frame)
        fut_data = self.get_fut_data(frame)

        valid_id = self.get_valid_id_pos_and_kp(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            # print(f"pre_data[0] is {pre_data[0]}, fut_data[0] is {fut_data[0]}, valid_id is {valid_id}")
            return None

        pre_motion, pre_motion_kp, pre_motion_mask = self.get_formatted_pre_data(pre_data, valid_id)
        fut_motion, fut_motion_kp, fut_motion_mask = self.get_formatted_fut_data(fut_data, valid_id)

        # curr_robot_data = self.get_relevant_robot_data(frame)

        data = {
            'pre_motion': pre_motion,
            'pre_motion_mask': pre_motion_mask,
            'pre_kp': pre_motion_kp,
            'pre_data': pre_data,
            'heading': None,
            'fut_motion': fut_motion,
            'fut_motion_mask': fut_motion_mask,
            'fut_kp': fut_motion_kp,
            'fut_data': fut_data,
            'traj_scale': self.traj_scale,
            'scene_map': self.geom_scene_map,
            'seq': self.seq_name,
            'frame': frame,
            'valid_id': valid_id,
            # 'robot_data': curr_robot_data,
        }

        return data

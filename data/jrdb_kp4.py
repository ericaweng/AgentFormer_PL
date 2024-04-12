""" dataloader for jrdb for positions, heading, hst-processed 3d_kp """

import torch
import numpy as np
from data import ped_interactions


class jrdb_preprocess(object):

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

        # trajectory positions information
        split_type = parser.get('split_type', 'full')
        assert split_type in ['full', 'egomotion', 'no_egomotion']  # only use hst odometry-adjusted data for this preprocessor

        # label_path = f'{data_root}/odometry_adjusted/{seq_name}.csv'
        path = f'{data_root}/../jrdb_adjusted/odometry_adjusted/{seq_name}_kp.npz'
        label_path = f'{data_root}/{seq_name}.txt'
        data = np.load(path, allow_pickle=True)['arr_0'].item()
        self.all_kp_data = data

        self.gt = np.genfromtxt(label_path, delimiter=' ', dtype=float)

        self.kp_mask = np.arange(33)

        # check that frame_ids are equally-spaced
        frames = np.unique(self.gt[:, 0].astype(int))
        frame_ids_diff = np.diff(frames)
        assert np.all(frame_ids_diff == frame_ids_diff[
            0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

        fr_start, fr_end = frames.min(), frames.max()
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
        self.class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6,
                            'Person': 7, 'Misc': 8, 'DontCare': 9, 'Traffic_cone': 10,
                            'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13,
                            'Bicycle': 14, 'Bus': 15, 'Trailer': 16, 'Emergency': 17, 'Construction': 18}
        # for row_index in range(len(self.gt)):
        #     self.gt[row_index][2] = class_names[self.gt[row_index][2]]
        self.gt = self.gt.astype('float32')
        self.xind, self.zind = 2,3
        self.heading_ind = 4

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self):
        return self.num_fr

    def get_data(self, frame_id):
        if frame_id in self.all_kp_data:
            data_kp = self.all_kp_data[frame_id]
        else:
            data_kp = {}

        data_pos = self.gt[self.gt[:, 0] == frame_id]
        return {'kp': data_kp, 'pos': data_pos}

    def get_pre_data(self, frame):
        DataList = []
        for i in range(self.past_frames):
            frame_id = frame - i * self.frame_skip
            DataList.append(self.get_data(frame_id))
        return DataList

    def get_fut_data(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            frame_id = frame + i * self.frame_skip
            DataList.append(self.get_data(frame_id))
        return DataList

    def get_valid_id_pos_and_kp(self, pre_data, fut_data):
        ''' only return is_valid=True if a ped has pos + 2d kp information
        only for which both pose and position annotation exist'''
        cur_ped_id = self.GetID(pre_data[0]['pos'])  # ped_ids this frame

        valid_id = []
        for idx in cur_ped_id:
            is_invalid = False
            for frame in pre_data[:self.min_past_frames]:
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:,1] or self.exclude_kpless_data and (
                        idx not in frame['kp'] or np.all(frame['kp'][idx]==0) or np.all(np.isnan(frame['kp'][idx]))):
                    is_invalid = True
                    break
            if is_invalid:
                continue
            for frame in fut_data[:self.min_future_frames]:
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:,1] or self.exclude_kpless_data and idx not in frame['kp']:
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
        heading = np.zeros((len(valid_id)))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data['pos'][cur_data['pos'][:, 1] == idx].squeeze()[self.heading_ind]
        return heading  # don't need transformed theta given data preprocessed w odometry

    def get_heading_avg(self, all_data, valid_id):
        heading = np.zeros((len(valid_id)))
        for i, idx in enumerate(valid_id):
            headings_this_ped = []
            for ts in range(len(all_data)):
                h = all_data[ts]['pos'][all_data[ts]['pos'][:, 1] == idx].squeeze()
                if len(h) == 0:
                    continue
                h = h[self.heading_ind]
                headings_this_ped.append((np.cos(h), np.sin(h)))
            heading_avg = np.stack(headings_this_ped).mean(0)
            heading[i] = np.arctan2(heading_avg[1], heading_avg[0])
        return heading  # don't need transformed theta given data preprocessed w odometry

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

                # edit frame index for pre data, which is ordered backwards
                if is_pre:
                    frame_i = num_frames - 1 - frame_i
                if not self.zero_kpless_data\
                        or self.zero_kpless_data and single_frame['kp'] is not None and ped_id in single_frame['kp']:
                    found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[
                                 [self.xind, self.zind]] / self.past_traj_scale
                    assert len(found_data) != 0
                    box_3d[frame_i] = torch.from_numpy(found_data).float()
                    mask_i[frame_i] = 1.0
                else:
                    import ipdb; ipdb.set_trace()
                    box_3d[frame_i] = torch.zeros((1, 2))
                    mask_i[frame_i] = 0
                # if joints info exists
                if single_frame['kp'] is not None and ped_id in single_frame['kp']:
                    kp_3d[frame_i] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                else:
                    kp_3d[frame_i] = 0
            motion.append(box_3d)
            mask.append(mask_i)
            # replace nan with 0
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

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (
        frame, self.TotalFrame())

        pre_data = self.get_pre_data(frame)
        fut_data = self.get_fut_data(frame)

        valid_id = self.get_valid_id_pos_and_kp(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        heading = self.get_heading(pre_data[0], valid_id)
        heading_avg = self.get_heading_avg(pre_data, valid_id)
        pred_mask = None

        pre_motion, pre_motion_kp, pre_motion_mask = self.get_formatted_pre_data(pre_data, valid_id)
        fut_motion, fut_motion_kp, fut_motion_mask = self.get_formatted_fut_data(fut_data, valid_id)
        # , pre_motion_kp_mask
        # , fut_motion_kp_mask

        data = {
                # These fields have a pre and fut, and are organized by ped. (num_peds, num_timesteps, **)
                'pre_motion': pre_motion,
                'pre_motion_mask': pre_motion_mask,
                'pre_kp': pre_motion_kp,
                # 'pre_kp_mask': pre_motion_kp_mask,
                'fut_motion': fut_motion,
                'fut_motion_mask': fut_motion_mask,
                'fut_kp': fut_motion_kp,
                # 'fut_kp_mask': fut_motion_kp_mask,
                'heading': heading,  # only the heading for the last obs timestep
                'heading_avg': heading_avg,  # the avg heading for all timesteps
                'traj_scale': self.traj_scale,
                'pred_mask': pred_mask,
                'scene_map': self.geom_scene_map,
                'seq': self.seq_name,
                'frame_scale': 1,
                'frame': frame,
                'valid_id': valid_id,
        }

        return data

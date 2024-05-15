"""
dataloader for jrdb for positions, heading, hst-processed 3d_kp
add config param to restrict data to include only agent-timesteps a certain dist away from robot
allow missing positions, heading, kp data to be recorded as 0, with mask 0
"""

import torch
import numpy as np
import pandas as pd
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
        assert split_type in ['full', 'hst_full', 'egomotion', 'no_egomotion']  # only use hst odometry-adjusted data for this preprocessor

        # label_path = f'{data_root}/odometry_adjusted/{seq_name}.csv'
        # if split == 'train':  # test deva's suggestion
        trajectories_path = f'{data_root}/{seq_name}.txt'
        # else:
        #     assert split in ['val', 'test']
        #     trajectories_path = f'{data_root}/../jrdb_adjusted/odometry_adjusted/{seq_name}.csv'

        self.gt = np.genfromtxt(trajectories_path, delimiter=' ', dtype=float)
        self.all_kp_data = np.load(f'{data_root}/agent_keypoints/{seq_name}_kp.npz', allow_pickle=True)['arr_0'].item()
        self.robot_data = np.genfromtxt(f'{data_root}/robot_poses/{seq_name}_robot.txt', delimiter=' ', dtype=float)

        self.kp_mask = np.arange(33)

        # check that frame_ids are equally-spaced
        gt_frames = np.unique(self.gt[:, 0].astype(int))
        frame_ids_diff = np.diff(gt_frames)
        assert np.all(frame_ids_diff == frame_ids_diff[
            0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

        # join frames array to robot data
        assert len(gt_frames) == len(self.robot_data), 'gt_frames and robot_data must have the same length'
        self.robot_data = np.concatenate([gt_frames[:,np.newaxis], self.robot_data], axis=1)

        # specify data split split
        fr_start, fr_end = gt_frames.min(), gt_frames.max()
        if 'half_and_half' in parser.get('split_type', 'normal'):
            train_size = len(gt_frames) * 3 // 4
            test_size = len(gt_frames) * 1 // 4
            if split == 'train':
                self.num_fr = train_size
                self.init_frame = fr_start
            else:
                self.num_fr = test_size
                self.init_frame = fr_start + train_size
        else:
            self.num_fr = len(gt_frames)
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

    def format_data(self, data, num_frames, valid_id, is_pre=False):
        motion = []
        mask = []
        headings = []
        kp_motion = []
        for ped_id in valid_id:
            mask_i = torch.zeros(num_frames)
            box_3d = torch.zeros([num_frames, 2])
            heading = torch.zeros([num_frames])
            kp_3d = torch.zeros([num_frames, len(self.kp_mask), 3])
            for f_i, frame_i in enumerate(range(num_frames)):
                single_frame = data[frame_i]
                pos_history = single_frame['pos']

                # edit frame index for pre data, which is ordered backwards
                if is_pre:
                    frame_i = num_frames - 1 - frame_i
                if (ped_id in single_frame['pos'][:,1]
                        and (not self.zero_kpless_data
                        or self.zero_kpless_data and single_frame['kp'] is not None and ped_id in single_frame['kp'])):
                    found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[
                                 [self.xind, self.zind]] / self.past_traj_scale
                    assert len(found_data) != 0
                    box_3d[frame_i] = torch.from_numpy(found_data).float()
                    heading[frame_i] = torch.from_numpy(np.array(single_frame['pos'][single_frame['pos'][:, 1] == ped_id].squeeze()[self.heading_ind]))
                    mask_i[frame_i] = 1.0
                else:
                    box_3d[frame_i] = torch.full((1, 2), 0)#torch.nan)  #
                    heading[frame_i] = 0
                    mask_i[frame_i] = 0.0
                # if joints info exists
                if single_frame['kp'] is not None and ped_id in single_frame['kp']:
                    kp_3d[frame_i] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                else:
                    kp_3d[frame_i] = 0#torch.nan
            motion.append(box_3d)
            mask.append(mask_i)
            headings.append(heading)
            # replace nan with 0
            kp_3d[kp_3d != kp_3d] = 0
            kp_motion.append(kp_3d)

        return motion, kp_motion, headings, mask

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

    def get_valid_ids(self, ids, pre_data, fut_data):
        # combine pre_data and fut_data and stack
        all_data = np.concatenate([d['pos'] for d in pre_data + fut_data])

        min_dist_from_robot = self.parser.get('min_dist_to_robot', 1000.0)
        # remove agents when their min distance from robot > min_dist_from_robot
        # group by timestep (col 0) and then find distance to the robot at the same timestep

        robot_data_df = pd.DataFrame(self.robot_data, columns=['frames', 'x', 'y', 'theta'])
        gt_df = pd.DataFrame(all_data, columns=['frame', 'id', 'x', 'y', 'theta'])

        # Merge and calculate the difference
        merged_df = pd.merge(gt_df, robot_data_df, left_on='frame', right_on='frames', suffixes=('_gt', '_robot'), how='inner')
        merged_df['x_diff'] = merged_df['x_gt'] - merged_df['x_robot']
        merged_df['y_diff'] = merged_df['y_gt'] - merged_df['y_robot']

        # group by timestep; find min dist to robot across timesteps; if min dist < min_dist_from_robot, keep agent; o/w remove agent and all associated timesteps for that agent.
        merged_df['dist_to_robot'] = np.linalg.norm(merged_df[['x_diff', 'y_diff']].values, axis=1)
        condition_by_ped = merged_df.groupby('id')['dist_to_robot'].min() < min_dist_from_robot

        filtered_ids = [ped_id for ped_id in ids if ped_id in condition_by_ped[condition_by_ped].index]

        return filtered_ids

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, (
                'frame is %d, total is %d' % (frame, self.TotalFrame()))

        pre_data = self.get_pre_data(frame)
        fut_data = self.get_fut_data(frame)

        all_ids = np.unique(np.concatenate([pd['pos'][:, 1] for pd in pre_data]))
        # assert len(valid_id) > 0
        # return None
        valid_id = self.get_valid_ids(all_ids, pre_data, fut_data)
        if len(valid_id) == 0:
            return None

        pre_motion, pre_motion_kp, pre_heading, pre_motion_mask = self.get_formatted_pre_data(pre_data, valid_id)
        fut_motion, fut_motion_kp, fut_heading, fut_motion_mask = self.get_formatted_fut_data(fut_data, valid_id)
        # assert none are nan

        heading_avg = [torch.mean(h) for h in pre_heading]
        assert heading_avg is not None

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
                'pre_heading': pre_heading,  # all available past headings
                'fut_heading': fut_heading,  # all available future headings
                'heading': None, #all_headings,  # all available past headings
                'heading_avg': heading_avg,  # the avg heading for all timesteps
                'traj_scale': self.traj_scale,
                'pred_mask': None,
                'scene_map': self.geom_scene_map,
                'seq': self.seq_name,
                'frame_scale': 1,
                'frame': frame,
                'valid_id': valid_id,
        }

        return data

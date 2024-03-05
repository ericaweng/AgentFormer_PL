""" dataloader for jrdb for positions + heading, 2d_kp, kp score, cam_extrinsics, cam_intrinsics """

import torch
import numpy as np
from data import ped_interactions


class jrdb_preprocess(object):

    def __init__(self, data_root, seq_name, parser, split='train', phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.exclude_cats = parser.get('exclude_cats', [])
        self.exclude_kpless_data = parser.get('exclude_kpless_data', False)
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

        # self.data_file = os.path.join(data_root, 'pedx_joint_pos2.npz')
        label_path = f'{data_root}/{seq_name}.txt'
        self.gt = np.genfromtxt(label_path, delimiter=' ', dtype=float)

        path = f'{data_root}/poses_2d.npz'
        self.data = np.load(path, allow_pickle=True)['arr_0'].item()

        self.all_kp_data = self.data['poses'][seq_name]
        self.all_score_data = self.data['scores'][seq_name]
        self.cam_ids = self.data['cam_ids'][seq_name]
        self.cam_extrinsics = self.data['extrinsics'][seq_name]
        self.cam_intrinsics = self.data['intrinsics'][seq_name]
        self.num_intrinsic_dims = 9
        self.num_extrinsic_dims = 7

        # total invalid peds... just see how many there are total
        self.total_invalid_peds = 0
        self.total_valid_peds = 0
        # self.all_head_heading_data = self.all_data['head_heading'][capture_date]
        # self.all_body_heading_data = self.all_data['body_heading'][capture_date]

        self.kp_mask = np.arange(17)

        # self.kp_mask = np.array(list(map(int, parser.kp_mask)))  # todo

        # check that frame_ids are equally-spaced
        frames = np.unique(self.gt[:, 0].astype(int))
        frame_ids_diff = np.diff(frames)
        assert np.all(frame_ids_diff == frame_ids_diff[
            0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

        fr_start, fr_end = frames.min(), frames.max()
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
            assert frame_id in self.all_score_data
            assert frame_id in self.cam_extrinsics
            assert frame_id in self.cam_intrinsics
            data_kp = self.all_kp_data[frame_id]
            data_score = self.all_score_data[frame_id]
            cam_ids = self.cam_ids[frame_id]
            cam_intrinsics = self.cam_intrinsics[frame_id]
            cam_extrinsics = self.cam_extrinsics[frame_id]
        else:
            data_kp = {}
            data_score = {}
            cam_ids = {}
            cam_intrinsics = {}
            cam_extrinsics = {}

        data_pos = self.gt[self.gt[:, 0] == frame_id]
        return {'kp': data_kp, 'pos': data_pos, 'score': data_score, 'cam_ids': cam_ids,
                         'cam_intrinsics': cam_intrinsics, 'cam_extrinsics': cam_extrinsics}

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
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:,1] or self.exclude_kpless_data and idx not in frame['kp']:
                    is_invalid = True
                    break
            if is_invalid:
                self.total_invalid_peds += 1
                continue
            for frame in fut_data[:self.min_future_frames]:
                if isinstance(frame['pos'], list) or idx not in frame['pos'][:,1] or self.exclude_kpless_data and idx not in frame['kp']:
                    is_invalid = True
                    break
            if is_invalid:
                self.total_invalid_peds += 1
                continue
            if len(self.exclude_cats) > 0:
                import ipdb; ipdb.set_trace()
                history_positions = np.array([p['pos'][p['pos'][:, 1] == idx][0][2:4] for p in pre_data])
                future_positions = np.array([p['pos'][p['pos'][:, 1] == idx][0][2:4] for p in fut_data])
                pos = np.concatenate([history_positions, future_positions], axis=0)
                if ped_interactions.is_static(pos)[0]:
                    continue
            valid_id.append(idx)
            self.total_valid_peds+=1

        return valid_id

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros((len(valid_id), 2))
        for i, idx in enumerate(valid_id):
            h = cur_data['pos'][cur_data['pos'][:, 1] == idx].squeeze()[self.heading_ind]
            heading[i] = np.array([-np.cos(h), np.sin(h)])
        return heading

    def get_heading_avg(self, all_data, valid_id):
        heading = np.zeros((len(all_data), len(valid_id), 2))
        for ts in range(len(all_data)):
            for i, idx in enumerate(valid_id):
                this_ped = all_data[ts]['pos'][:, 1] == idx
                this_ped_row = all_data[ts]['pos'][this_ped]
                if len(all_data[ts]['pos'][this_ped]) == 0:
                    continue
                h = this_ped_row.squeeze()[self.heading_ind]
                heading[ts,i] = -np.cos(h), np.sin(h)
        return heading.mean(0)

    def format_data(self, data, num_frames, valid_id, is_pre=False):
        motion = []
        mask = []
        kp_motion = []
        scores = []
        cam_ids = []
        cam_intrinsics = []
        cam_extrinsics = []
        for ped_id in valid_id:
            mask_i = torch.zeros(num_frames)
            box_3d = torch.zeros([num_frames, 2])
            kp_3d = torch.zeros([num_frames, len(self.kp_mask), 2])
            score = torch.zeros([num_frames, len(self.kp_mask)])
            cam_id = torch.zeros([num_frames])
            cam_intrinsic = torch.zeros([num_frames, self.num_intrinsic_dims])
            cam_extrinsic = torch.zeros([num_frames, self.num_extrinsic_dims])
            for f_i, frame_i in enumerate(range(num_frames)):
                single_frame = data[frame_i]
                pos_history = single_frame['pos']

                if is_pre:
                    frame_i = num_frames - 1 - frame_i
                if is_pre and f_i > self.past_frames_pos:
                    import ipdb; ipdb.set_trace()
                    box_3d[frame_i] = 0
                    mask_i[frame_i] = 0.0
                else:
                    found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[
                                     [self.xind, self.zind]] / self.past_traj_scale
                    assert len(found_data) != 0
                    box_3d[frame_i] = torch.from_numpy(found_data).float()
                    mask_i[frame_i] = 1.0
                if single_frame['kp'] is not None and ped_id in single_frame['kp']:
                    kp_3d[frame_i] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                    score[frame_i] = torch.from_numpy(single_frame['score'][ped_id]).float()
                    cam_id[frame_i] = single_frame['cam_ids'][ped_id]
                    cam_intrinsic[frame_i] = torch.from_numpy(single_frame['cam_intrinsics'][ped_id])
                    cam_extrinsic[frame_i] = torch.from_numpy(single_frame['cam_extrinsics'][ped_id])
                else:
                    kp_3d[frame_i] = 0
                    score[frame_i] = 0
                    cam_id[frame_i] = -1
                    cam_intrinsic[frame_i] = 0
                    cam_extrinsic[frame_i] = 0
            motion.append(box_3d)
            mask.append(mask_i)
            kp_motion.append(kp_3d)
            scores.append(score)
            cam_ids.append(cam_id)
            cam_intrinsics.append(cam_intrinsic)
            cam_extrinsics.append(cam_extrinsic)

        return motion, kp_motion, mask, scores, cam_ids, cam_intrinsics, cam_extrinsics

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

        # interaction_mask = self.get_interaction_mask(pre_data, fut_data, valid_id)

        heading = self.get_heading(pre_data[0], valid_id)
        heading_avg = self.get_heading_avg(pre_data, valid_id)
        pred_mask = None

        (pre_motion, pre_motion_kp, pre_motion_mask, pre_scores, pre_cam_ids,
         pre_cam_intrinsics, pre_cam_extrinsics) = self.get_formatted_pre_data(pre_data, valid_id)
        (fut_motion, fut_motion_kp, fut_motion_mask, fut_scores, fut_cam_ids,
         fut_cam_intrinsics, fut_cam_extrinsics) = self.get_formatted_fut_data(fut_data, valid_id)

        data = {
                'pre_motion': pre_motion,
                'pre_motion_mask': pre_motion_mask,
                'pre_kp': pre_motion_kp,
                'pre_kp_scores': pre_scores,
                'pre_cam_id': pre_cam_ids,
                'pre_cam_intrinsics': pre_cam_intrinsics,
                'pre_cam_extrinsics': pre_cam_extrinsics,
                'fut_motion': fut_motion,
                'fut_motion_mask': fut_motion_mask,
                'history_data': pre_data,
                'future_data': fut_data,
                'fut_kp': fut_motion_kp,
                'fut_kp_scores': fut_scores,
                'fut_cam_id': fut_cam_ids,
                'fut_cam_intrinsics': fut_cam_intrinsics,
                'fut_cam_extrinsics': fut_cam_extrinsics,
                # 'interaction_mask': interaction_mask,
                'heading': heading,  # only the heading for the last obs timestep
                'heading_avg': heading_avg,  # the avg heading for all timesteps
                'traj_scale': self.traj_scale,
                'pred_mask': pred_mask,
                'scene_map': self.geom_scene_map,
                'seq': self.seq_name,
                'frame_scale': 1,
                'frame': frame,
                'image_paths': [f'{self.seq_name}/{f:06d}.jpg' for f in range(frame-self.past_frames, frame+self.future_frames)],
                'valid_id': valid_id,
        }

        return data

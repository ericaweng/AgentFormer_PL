""" dataloader for jrdb for positions + heading, 2d_kp, scores """

import torch
import numpy as np


class jrdb_preprocess(object):

    def __init__(self, data_root, seq_name, parser, split='train', phase='training'):
        self.parser = parser
        self.dataset = parser.dataset
        self.data_root = data_root
        self.past_frames = parser.past_frames
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

        path = f'{data_root}/poses_stitched_2d.npz'
        self.pose_and_cam_data = np.load(path, allow_pickle=True)[seq_name].item()  # self.all_data['kp'][seq_name]

        self.all_kp_data = {}
        self.all_score_data = {}
        for frame in self.pose_and_cam_data:
            self.all_kp_data[frame] = {}
            self.all_score_data[frame] = {}
            for ped_id in self.pose_and_cam_data[frame]:
                self.all_kp_data[frame][ped_id] = self.pose_and_cam_data[frame][ped_id]['pose']
                self.all_score_data[frame][ped_id] = self.pose_and_cam_data[frame][ped_id]['score']

        # self.all_cam_data = {}
        # for frame in self.pose_and_cam_data:
        #     if frame not in self.all_cam_data:
        #         self.all_cam_data[frame] = {}
        #     for ped_id in self.pose_and_cam_data[frame]:
        #         self.all_cam_data[frame][ped_id] = self.pose_and_cam_data[frame][ped_id]['intrinsics']

        # self.all_trajs_data = self.all_data['pos'][capture_date]
        # self.all_head_heading_data = self.all_data['head_heading'][capture_date]
        # self.all_body_heading_data = self.all_data['body_heading'][capture_date]

        self.kp_mask = np.arange(17)
        # self.kp_mask = np.array(list(map(int, parser.kp_mask)))  # todo
        # self.num_kp = len(self.kp_mask)
        # check that frame_ids are equally-spaced
        # frames = sorted(map(int, self.all_kp_data.keys()))
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

    def PreData(self, frame):
        """history is backwards"""
        position_data = []
        for i in range(self.past_frames):
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            position_data.append(data)
        return position_data

    def FutureData(self, frame):
        position_data = []
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            position_data.append(data)
        return position_data

    def PreData_pos_and_kp(self, frame):
        DataList = []
        for i in range(self.past_frames):
            frame_id = frame - i * self.frame_skip
            if frame_id in self.all_kp_data:
                data_kp = self.all_kp_data[frame_id]
            else:
                data_kp = {}
            if frame_id in self.all_score_data:
                data_score = self.all_score_data[frame - i * self.frame_skip]
            else:
                data_score = {}
            data_pos = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            DataList.append({'kp': data_kp, 'pos': data_pos, 'score': data_score})
        return DataList

    def FutureData_pos_and_kp(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            frame_id = frame + i * self.frame_skip
            if frame_id in self.all_kp_data:
                data_kp = self.all_kp_data[frame_id]
            else:
                data_kp = {}
            if frame_id in self.all_score_data:
                data_score = self.all_score_data[frame + i * self.frame_skip]
            else:
                data_score = {}
            data_pos = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append({'kp': data_kp, 'pos': data_pos, 'score': data_score})
        return DataList

    def get_valid_id_pos_and_kp(self, pre_data, fut_data):
        cur_ped_id = self.GetID(pre_data[0]['pos'])  # ped_ids this frame
        valid_id = []
        for idx in cur_ped_id:
            is_invalid = False
            for frame in pre_data[:self.min_past_frames]:
                if isinstance(frame, list):
                    import ipdb; ipdb.set_trace()
                elif idx not in frame['kp']:
                    is_invalid = True
                    break
                elif idx not in frame['pos']:
                    is_invalid = True
                    break
            for frame in fut_data[:self.min_future_frames]:
                if isinstance(frame, list):
                    import ipdb; ipdb.set_trace()
                elif idx not in frame['kp']:
                    is_invalid = True
                    break
                elif idx not in frame['pos']:
                    is_invalid = True
                    break
            if not is_invalid:
                valid_id.append(idx)
        return valid_id

    def get_valid_id(self, pre_data, fut_data):
        cur_ped_id = self.GetID(pre_data[0]['pos'])  # ped_ids this frame
        valid_id = []
        for idx in cur_ped_id:
            exist_pre = [(False if isinstance(frame, list) else (idx in frame[:, 1])) for frame in
                         pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(frame, list) else (idx in frame[:, 1])) for frame in
                         fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data['pos'][cur_data['pos'][:, 1] == idx].squeeze()[self.heading_ind]
        return heading

    def get_heading_avg(self, all_data, valid_id):
        heading = np.zeros((len(all_data), len(valid_id), 2))
        for ts in range(len(all_data)):
            for i, idx in enumerate(valid_id):
                h = all_data[ts]['pos'][all_data[ts]['pos'][:, 1] == idx].squeeze()[self.heading_ind]
                heading[ts,i] = np.cos(h), np.sin(h)
                import ipdb; ipdb.set_trace()
        return heading.mean(0)

    def PreJoints(self, history, valid_id):
        kp = []
        scores = []
        for ped_id in valid_id:
            kp_3d = torch.zeros([self.past_frames, len(self.kp_mask), 2])
            score = torch.zeros([self.past_frames, len(self.kp_mask)])
            for frame_i in range(self.past_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    if ped_id in single_frame['kp']:
                        kp_3d[self.past_frames-1 - frame_i, :, :] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                        score[self.past_frames - 1 - frame_i, :] = torch.from_numpy(
                                single_frame['score'][ped_id]).float()
                    else:
                        kp_3d[self.past_frames-1 - frame_i, :, :] = kp_3d[self.past_frames - frame_i, :, :]
                        score[self.past_frames - 1 - frame_i, :] = 0.0
                elif frame_i > 0:
                    kp_3d[self.past_frames-1 - frame_i, :, :] = kp_3d[self.past_frames - frame_i, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            kp.append(kp_3d)
            scores.append(score)

        return kp, scores

    def FutureJoints(self, history, valid_id):
        kp = []
        scores = []
        for ped_id in valid_id:
            score = torch.zeros([self.past_frames, 1])
            kp_3d = torch.zeros([self.future_frames, len(self.kp_mask), 2])
            for frame_i in range(self.future_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    kp_3d[frame_i, :, :] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                    if ped_id in single_frame['kp']:
                        kp_3d[frame_i, :, :] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                        score[frame_i, :] = torch.from_numpy(
                                single_frame['score'][ped_id]).float()
                    else:
                        kp_3d[frame_i, :, :] = kp_3d[self.past_frames - frame_i, :, :]
                        score[frame_i, :] = 0.0
                elif frame_i > 0:  # if the ped doesn't exist, then just copy previous frame? will it be masked out?
                    kp_3d[frame_i, :, :] = kp_3d[frame_i - 1, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            kp.append(kp_3d)
        return kp, scores

    def PreMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            for j in range(self.past_frames):
                past_data = DataTuple[j]  # past_data
                if len(past_data) > 0 and identity in past_data[:, 1]:
                    found_data = past_data[past_data[:, 1] == identity].squeeze()[
                                     [self.xind, self.zind]] / self.past_traj_scale
                    box_3d[self.past_frames - 1 - j, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames - 1 - j] = 1.0
                elif j > 0:
                    box_3d[self.past_frames - 1 - j, :] = box_3d[self.past_frames - j, :]  # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
        return motion, mask

    def FutureMotion(self, DataTuple, valid_id):
        motion = []
        mask = []
        for identity in valid_id:
            mask_i = torch.zeros(self.future_frames)
            pos_3d = torch.zeros([self.future_frames, 2])
            for j in range(self.future_frames):
                fut_data = DataTuple[j]  # cur_data
                if len(fut_data) > 0 and identity in fut_data[:, 1]:
                    found_data = fut_data[fut_data[:, 1] == identity].squeeze()[
                                     [self.xind, self.zind]] / self.traj_scale
                    pos_3d[j, :] = torch.from_numpy(found_data).float()
                    mask_i[j] = 1.0
                elif j > 0:
                    pos_3d[j, :] = pos_3d[j - 1, :]  # if none, copy from previous
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(pos_3d)
            mask.append(mask_i)
        return motion, mask

    def PreMotionJoints(self, history, valid_id):
        motion = []
        mask = []
        kp_motion = []
        scores = []
        for ped_id in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            kp_3d = torch.zeros([self.past_frames, len(self.kp_mask), 2])
            score = torch.zeros([self.past_frames, len(self.kp_mask)])
            for frame_i in range(self.past_frames):
                single_frame = history[frame_i]
                assert len(single_frame['pos']) > 0 and ped_id in single_frame['pos'][:, 1], 'ped_id %d not found in frame %d' % (ped_id, frame_i)
                assert len(single_frame['kp'][ped_id]) > 0, 'ped_id %d not found in frame %d' % (ped_id, frame_i)
                pos_history = single_frame['pos']
                found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[
                                 [self.xind, self.zind]] / self.past_traj_scale
                box_3d[self.past_frames - 1 - frame_i, :] = torch.from_numpy(found_data).float()
                mask_i[self.past_frames - 1 - frame_i] = 1.0
                kp_3d[self.past_frames - 1 - frame_i, :, :] = torch.from_numpy(
                        single_frame['kp'][ped_id]).float()
                score[self.past_frames - 1 - frame_i, :] = torch.from_numpy(
                        single_frame['score'][ped_id]).float()
            motion.append(box_3d)
            mask.append(mask_i)
            kp_motion.append(kp_3d)
            scores.append(score)
        return motion, kp_motion, mask, scores

    def FutureMotionJoints(self, history, valid_id):
        motion = []
        mask = []
        joints = []
        scores = []
        for ped_id in valid_id:
            score = torch.zeros([self.future_frames, len(self.kp_mask)])
            mask_i = torch.zeros(self.future_frames)
            box_3d = torch.zeros([self.future_frames, 2])
            kp_3d = torch.zeros([self.future_frames, len(self.kp_mask), 2])
            for frame_i in range(self.future_frames):
                single_frame = history[frame_i]
                assert len(single_frame['pos']) > 0 and ped_id in single_frame[
                    'pos'][:, 1], 'ped_id %d not found in frame %d' % (ped_id, frame_i)
                assert len(single_frame['kp'][ped_id]) > 0, 'ped_id %d not found in frame %d' % (ped_id, frame_i)
                pos_history = single_frame['pos']
                found_data = pos_history[pos_history[:, 1] == ped_id].squeeze()[
                                 [self.xind, self.zind]] / self.past_traj_scale
                box_3d[frame_i, :] = torch.from_numpy(found_data).float()
                mask_i[frame_i] = 1.0
                kp_3d[frame_i, :, :] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                kp_3d[frame_i, :, :] = torch.from_numpy(single_frame['kp'][ped_id]).float()
                score[frame_i, :] = torch.from_numpy(
                        single_frame['score'][ped_id]).float()
            motion.append(box_3d)
            mask.append(mask_i)
            joints.append(kp_3d)
            scores.append(score)
        return motion, joints, mask, scores

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (
        frame, self.TotalFrame())

        pre_data = self.PreData_pos_and_kp(frame)
        fut_data = self.FutureData_pos_and_kp(frame)

        valid_id = self.get_valid_id_pos_and_kp(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        heading = self.get_heading(pre_data[0], valid_id)
        heading_avg = self.get_heading_avg(pre_data, valid_id)
        pred_mask = None

        pre_motion, pre_motion_kp, pre_motion_mask, scores = self.PreMotionJoints(pre_data, valid_id)
        fut_motion, fut_motion_kp, fut_motion_mask, scores = self.FutureMotionJoints(fut_data, valid_id)

        data = {
                'pre_motion': pre_motion,
                'pre_kp': pre_motion_kp,
                'fut_motion': fut_motion,
                'fut_kp': fut_motion_kp,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                # 'pre_data': pre_data,
                # 'fut_data': fut_data,
                'heading': heading,  # only the heading for the last obs timestep
                'heading_avg': heading_avg,  # the avg heading for all timesteps
                'valid_id': valid_id,
                'traj_scale': self.traj_scale,
                'pred_mask': pred_mask,
                'scene_map': self.geom_scene_map,
                # 'scene_vis_map': self.scene_vis_map,
                'seq': self.seq_name,
                'frame_scale': 1,
                'frame': frame,
        }

        return data

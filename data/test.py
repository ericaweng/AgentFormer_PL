import torch, os, numpy as np, copy
import cv2
from .map import GeometricMap


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

        self.pose_and_cam_data = np.load(path, allow_pickle=True)[seq_name].item()  # self.all_data['joints'][seq_name]

        self.all_joints_data = {}
        self.all_score_data = {}
        for frame in self.pose_and_cam_data:
            self.all_joints_data[frame] = {}
            self.all_score_data[frame] = {}
            for ped_id in self.pose_and_cam_data[frame]:
                self.all_joints_data[frame][ped_id] = self.pose_and_cam_data[frame][ped_id]['pose']
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

        self.COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12],
                                         [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]
        self.joints_mask = np.arange(17)  # ([0,1,2,3,4,5,6,7,8,9,10,11,12,16,18,19,20,26,27,28])-1  # 19 joints
        # self.joints_mask = np.array(list(map(int, parser.joints_mask)))  # todo
        # self.num_joints = len(self.joints_mask)
        # check that frame_ids are equally-spaced
        # frames = sorted(map(int, self.all_joints_data.keys()))
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
        self.xind, self.zind = 2, 3

    def GetID(self, data):
        id = []
        for ped_id in data:
            id.append(ped_id)
        return id

    def TotalFrame(self):
        return self.num_fr

    def PreData(self, frame):
        DataList = []
        for i in range(self.past_frames):
            frame_id = frame - i * self.frame_skip
            if frame_id in self.all_joints_data:
                data_joints = self.all_joints_data[frame_id]
            else:
                data_joints = {}
            if frame_id in self.all_score_data:
                data_score = self.all_score_data[frame - i * self.frame_skip]
            else:
                data_score = {}
            data_pos = self.all_trajs_data[frame - i * self.frame_skip]
            DataList.append({'joints': data_joints, 'pos': data_pos, 'score': data_score})
        return DataList

    def FutureData(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            frame_id = frame + i * self.frame_skip
            if frame_id in self.all_joints_data:
                data_joints = self.all_joints_data[frame_id]
            else:
                data_joints = {}
            if frame_id in self.all_score_data:
                data_score = self.all_score_data[frame + i * self.frame_skip]
            else:
                data_score = {}
            data_pos = self.all_trajs_data[frame + i * self.frame_skip][:2]
            heading = self.all_trajs_data[frame + i * self.frame_skip][2]
            import ipdb;
            ipdb.set_trace()
            DataList.append({'joints': data_joints, 'pos': data_pos, 'score': data_score, 'heading': heading})
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        """ only add peds who exist in all frames in min_past_frames"""
        peds_this_frame = self.GetID(pre_data[0]['pos'])
        valid_ped_ids = []
        for ped_id in peds_this_frame:
            exist_pre = [(False if isinstance(frame, list) else (ped_id in frame['pos'])) for frame in
                         pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(frame, list) else (ped_id in frame['pos'])) for frame in
                         fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_ped_ids.append(ped_id)
        return valid_ped_ids

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros((len(valid_id), 2))
        for i, ped_id in enumerate(valid_id):
            heading[i] = cur_data['heading'][ped_id]
        return heading

    def get_heading_avg(self, all_data, valid_id):
        heading = np.zeros((len(all_data), len(valid_id), 2))
        for ts in range(len(all_data)):
            for i, idx in enumerate(valid_id):
                h = all_data[ts][all_data[ts][:, 1] == idx].squeeze()[-1]
                heading[ts, i] = np.cos(h), np.sin(h)
        return heading.mean(0)

    def PreMotion(self, history, valid_id):
        motion = []
        mask = []
        joints_motion = []
        for ped_id in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            joints_3d = torch.zeros([self.past_frames, len(self.joints_mask), 2])
            score = torch.zeros([self.past_frames, len(self.joints_mask)])
            for frame_i in range(self.past_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    found_data = single_frame['pos'][ped_id] / self.past_traj_scale
                    box_3d[self.past_frames - 1 - frame_i, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames - 1 - frame_i] = 1.0
                    if ped_id in single_frame['joints']:
                        joints_3d[self.past_frames - 1 - frame_i, :, :] = torch.from_numpy(
                                single_frame['joints'][ped_id]).float()
                        score[self.past_frames - 1 - frame_i, :] = torch.from_numpy(
                                single_frame['score'][ped_id]).float()
                    else:
                        joints_3d[self.past_frames - 1 - frame_i, :, :] = joints_3d[self.past_frames - frame_i, :, :]
                        score[self.past_frames - 1 - frame_i, :] = 0.0
                elif frame_i > 0:
                    box_3d[self.past_frames - 1 - frame_i, :] = box_3d[self.past_frames - frame_i,
                                                                :]  # if none, copy from previous
                    joints_3d[self.past_frames - 1 - frame_i, :, :] = joints_3d[self.past_frames - frame_i, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
            joints_motion.append(joints_3d)
        return motion, joints_motion, mask

    def FutureMotion(self, history, valid_id):
        motion = []
        mask = []
        joints = []
        score = torch.zeros([self.past_frames, 1])
        for ped_id in valid_id:
            mask_i = torch.zeros(self.future_frames)
            box_3d = torch.zeros([self.future_frames, 2])
            joints_3d = torch.zeros([self.future_frames, len(self.joints_mask), 2])
            for frame_i in range(self.future_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    found_data = single_frame['pos'][ped_id] / self.traj_scale
                    box_3d[frame_i, :] = torch.from_numpy(found_data).float()
                    mask_i[frame_i] = 1.0
                    joints_3d[frame_i, :, :] = torch.from_numpy(single_frame['joints'][ped_id]).float()
                    if ped_id in single_frame['joints']:
                        joints_3d[frame_i, :, :] = torch.from_numpy(single_frame['joints'][ped_id]).float()
                        score[frame_i, :] = torch.from_numpy(
                                single_frame['score'][ped_id]).float()
                    else:
                        joints_3d[frame_i, :, :] = joints_3d[self.past_frames - frame_i, :, :]
                        score[frame_i, :] = 0.0
                elif frame_i > 0:  # if the ped doesn't exist, then just copy previous frame? will it be masked out?
                    box_3d[frame_i, :] = box_3d[frame_i - 1, :]
                    joints_3d[frame_i, :, :] = joints_3d[frame_i - 1, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
            joints.append(joints_3d)
        return motion, joints, mask

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (
                frame, self.TotalFrame())

        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        # pred_mask = self.get_pred_mask(pre_data[0], valid_id)
        heading = self.get_heading(pre_data[0], valid_id)
        heading_avg = self.get_heading_avg(pre_data, valid_id)
        pred_mask = None

        # pre_motion, pre_motion_mask = self.PreMotion(pre_data, valid_id)  # reverses history
        # fut_motion, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        pre_motion, pre_motion_joints, pre_motion_mask = self.PreMotion(pre_data, valid_id)
        fut_motion, fut_motion_joints, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {
                'pre_motion': pre_motion,
                'pre_joints': pre_motion_joints,
                'fut_motion': fut_motion,
                'fut_joints': fut_motion_joints,
                'fut_motion_mask': fut_motion_mask,
                'pre_motion_mask': pre_motion_mask,
                'pre_data': pre_data,
                'fut_data': fut_data,
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

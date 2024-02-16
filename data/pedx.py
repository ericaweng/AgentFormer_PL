import torch, os, numpy as np, copy
import cv2
from .map import GeometricMap


class PedXPreprocess(object):

    def __init__(self, data_root, capture_date, parser, split='train', phase='training'):
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
        self.capture_date = capture_date
        self.split = split
        self.phase = phase

        self.data_file = os.path.join(data_root, 'pedx_joint_pos2.npz')
        self.all_data = np.load(self.data_file, allow_pickle=True)[split].item()
        self.all_kp_data = self.all_data['joints'][capture_date]
        self.all_trajs_data = self.all_data['pos'][capture_date]  # [frame_id][ped_id] = [x, y, z]
        self.all_head_heading_data = self.all_data['head_heading'][capture_date]
        self.all_body_heading_data = self.all_data['body_heading'][capture_date]

        self.kp_mask = np.arange(24)#([0,1,2,3,4,5,6,7,8,9,10,11,12,16,18,19,20,26,27,28])-1  # 19 joints
        # self.kp_mask = np.array(list(map(int, parser.kp_mask)))  # todo
        self.num_joints = len(self.kp_mask)
        # check that frame_ids are equally-spaced
        frames = sorted(map(int, self.all_kp_data.keys()))
        frame_ids_diff = np.diff(frames)
        assert np.all(frame_ids_diff == frame_ids_diff[
            0]), f"frame_ids_diff ({frame_ids_diff}) must be equal to frame_ids_diff[0] ({frame_ids_diff[0]})"

        frame_ids = np.array(frames)
        fr_start, fr_end = frame_ids.min(), frame_ids.max()
        self.init_frame = fr_start
        num_fr = fr_end + 1 - fr_start
        self.num_fr = len(frames)
        assert num_fr == self.num_fr, f"num_fr ({num_fr}) must be equal to self.num_fr ({self.num_fr})"

        self.geom_scene_map = None
        # self.gt = self.gt.astype('float32')

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
            data_joints = self.all_kp_data[str(frame - i * self.frame_skip).zfill(7)]
            data_pos = self.all_trajs_data[str(frame - i * self.frame_skip).zfill(7)]
            data_body_heading = self.all_body_heading_data[str(frame - i * self.frame_skip).zfill(7)]
            data_head_heading = self.all_head_heading_data[str(frame - i * self.frame_skip).zfill(7)]
            DataList.append({'joints': data_joints, 'pos': data_pos,
                             'head_heading': data_head_heading, 'body_heading': data_body_heading})
        return DataList

    def FutureData(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            data_joints = self.all_kp_data[str(frame + i * self.frame_skip).zfill(7)]
            data_pos = self.all_trajs_data[str(frame + i * self.frame_skip).zfill(7)]
            DataList.append({'joints': data_joints, 'pos': data_pos})
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        """ only add peds who exist in all frames in min_past_frames"""
        peds_this_frame = self.GetID(pre_data[0]['joints'])
        valid_ped_ids = []
        for ped_id in peds_this_frame:
            exist_pre = [(False if isinstance(frame, list) else (ped_id in frame['joints'])) for frame in pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(frame, list) else (ped_id in frame['joints'])) for frame in fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_ped_ids.append(ped_id)
        return valid_ped_ids

    def get_pred_mask(self, cur_data, valid_id):
        pred_mask = np.zeros(len(valid_id), dtype=np.int)
        for i, ped_id in enumerate(valid_id):  # for each ped, get all the
            pred_mask[i] = np.array([joints for ped_id, joints in sorted(cur_data.items())])[cur_data.keys() == ped_id].squeeze()[-1]
        return pred_mask

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros((len(valid_id), 2))
        for i, ped_id in enumerate(valid_id):
            heading[i] = cur_data['body_heading'][ped_id]
        return heading

    def PreMotion(self, history, valid_id):
        motion = []
        mask = []
        kp_motion = []
        for ped_id in valid_id:
            mask_i = torch.zeros(self.past_frames)
            box_3d = torch.zeros([self.past_frames, 2])
            kp_3d = torch.zeros([self.past_frames, len(self.kp_mask), 3])
            for frame_i in range(self.past_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    found_data = single_frame['pos'][ped_id] / self.past_traj_scale
                    box_3d[self.past_frames-1 - frame_i, :] = torch.from_numpy(found_data).float()
                    mask_i[self.past_frames-1 - frame_i] = 1.0
                    kp_3d[self.past_frames-1 - frame_i, :, :] = torch.from_numpy(single_frame['joints'][ped_id]).float()
                elif frame_i > 0:
                    box_3d[self.past_frames-1 - frame_i, :] = box_3d[self.past_frames - frame_i, :]    # if none, copy from previous
                    kp_3d[self.past_frames-1 - frame_i, :, :] = kp_3d[self.past_frames - frame_i, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
            kp_motion.append(kp_3d)
        return motion, kp_motion, mask

    def FutureMotion(self, history, valid_id):
        motion = []
        mask = []
        joints = []
        for ped_id in valid_id:
            mask_i = torch.zeros(self.future_frames)
            box_3d = torch.zeros([self.future_frames, 2])
            kp_3d = torch.zeros([self.future_frames, len(self.kp_mask), 3])
            for frame_i in range(self.future_frames):
                single_frame = history[frame_i]
                if len(single_frame['pos'][ped_id]) > 0 and ped_id in single_frame['pos']:
                    found_data = single_frame['pos'][ped_id] / self.traj_scale
                    box_3d[frame_i, :] = torch.from_numpy(found_data).float()
                    mask_i[frame_i] = 1.0
                    kp_3d[frame_i, :, :] = torch.from_numpy(single_frame['joints'][ped_id]).float()
                elif frame_i > 0:  # if the ped doesn't exist, then just copy previous frame? will it be masked out?
                    box_3d[frame_i, :] = box_3d[frame_i - 1, :]
                    kp_3d[frame_i, :, :] = kp_3d[frame_i - 1, :, :]
                else:
                    raise ValueError('current id missing in the first frame!')
            motion.append(box_3d)
            mask.append(mask_i)
            joints.append(kp_3d)
        return motion, joints, mask

    def __call__(self, frame):

        assert frame - self.init_frame >= 0 and frame - self.init_frame <= self.TotalFrame() - 1, 'frame is %d, total is %d' % (frame, self.TotalFrame())


        pre_data = self.PreData(frame)
        fut_data = self.FutureData(frame)
        valid_id = self.get_valid_id(pre_data, fut_data)
        if len(pre_data[0]) == 0 or len(fut_data[0]) == 0 or len(valid_id) == 0:
            return None

        # if self.dataset == 'nuscenes_pred':
        # pred_mask = self.get_pred_mask(pre_data[0], valid_id)
        heading = self.get_heading(pre_data[0], valid_id)

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
            'heading': heading,
            'valid_id': valid_id,
            'traj_scale': self.traj_scale,
            'pred_mask': None, # pred_mask,
            'scene_map': self.geom_scene_map,
            'seq': self.capture_date,
            'frame_scale': 1,
            'frame': frame,
        }
        # print("pre_motion.shape:", pre_motion[0].shape)
        # print("len(pre_motion):", len(pre_motion))
        # print("fut_motion.shape:", fut_motion[0].shape)
        # print("len(fut_motion):", len(fut_motion))
        # print("fut_motion_mask.shape:", fut_motion_mask[0].shape)
        # print("len(fut_motion_mask):", len(fut_motion_mask))
        # print("pre_motion_mask.shape:", pre_motion_mask[0].shape)
        # print("len(pre_motion_mask):", len(pre_motion_mask))
        # print("pre_data.shape:", pre_data[0].shape)
        # print("len(pre_data):", len(pre_data))
        # print("fut_data.shape:", fut_data[0].shape)
        # if True:
        #     print("len(fut_data):", len(fut_data))
        #     print("valid_id.shape:", valid_id[0].shape)
        #     print("len(valid_id):", len(valid_id))

        return data

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
        self.gt = np.genfromtxt(label_path, delimiter=' ', dtype=str)

        # self.all_joints_data = self.all_data['joints'][capture_date]
        # self.all_trajs_data = self.all_data['pos'][capture_date]
        # self.all_head_heading_data = self.all_data['head_heading'][capture_date]
        # self.all_body_heading_data = self.all_data['body_heading'][capture_date]

        # self.joints_mask = np.arange(24)#([0,1,2,3,4,5,6,7,8,9,10,11,12,16,18,19,20,26,27,28])-1  # 19 joints
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
        self.class_names = class_names = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Truck': 4, 'Van': 5, 'Tram': 6,
                                          'Person': 7, 'Misc': 8, 'DontCare': 9, 'Traffic_cone': 10,
                                          'Construction_vehicle': 11, 'Barrier': 12, 'Motorcycle': 13,
                                          'Bicycle': 14, 'Bus': 15, 'Trailer': 16, 'Emergency': 17, 'Construction': 18}
        # for row_index in range(len(self.gt)):
        #     self.gt[row_index][2] = class_names[self.gt[row_index][2]]
        self.gt = self.gt.astype('float32')
        self.xind, self.zind = 2,3

    def GetID(self, data):
        id = []
        for i in range(data.shape[0]):
            id.append(data[i, 1].copy())
        return id

    def TotalFrame(self):
        return self.num_fr

    def PreData(self, frame):
        """history is backwards"""
        DataList = []
        for i in range(self.past_frames):
            if frame - i < self.init_frame:
                data = []
            data = self.gt[self.gt[:, 0] == (frame - i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def FutureData(self, frame):
        DataList = []
        for i in range(1, self.future_frames + 1):
            data = self.gt[self.gt[:, 0] == (frame + i * self.frame_skip)]
            DataList.append(data)
        return DataList

    def get_valid_id(self, pre_data, fut_data):
        cur_ped_id = self.GetID(pre_data[0])  # ped_ids this frame
        valid_id = []
        for idx in cur_ped_id:
            exist_pre = [(False if isinstance(frame, list) else (idx in frame[:, 1])) for frame in
                         pre_data[:self.min_past_frames]]
            exist_fut = [(False if isinstance(frame, list) else (idx in frame[:, 1])) for frame in
                         fut_data[:self.min_future_frames]]
            if np.all(exist_pre) and np.all(exist_fut):
                valid_id.append(idx)
        return valid_id

    def get_pred_mask(self, cur_data, valid_id):
        import ipdb; ipdb.set_trace()
        pred_mask = np.zeros(len(valid_id), dtype=np.int)
        for i, idx in enumerate(valid_id):  # for each valid ped, get the data of the last frame
            pred_mask[i] = cur_data[cur_data[:, 1] == idx].squeeze()[-1]
        return pred_mask

    def get_heading(self, cur_data, valid_id):
        heading = np.zeros(len(valid_id))
        for i, idx in enumerate(valid_id):
            heading[i] = cur_data[cur_data[:, 1] == idx].squeeze()[-1]
        return heading

    def get_heading_avg(self, all_data, valid_id):
        heading = np.zeros((len(all_data), len(valid_id), 2))
        for ts in range(len(all_data)):
            for i, idx in enumerate(valid_id):
                h = all_data[ts][all_data[ts][:, 1] == idx].squeeze()[-1]
                heading[ts,i] = np.cos(h), np.sin(h)
        return heading.mean(0)

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

        pre_motion, pre_motion_mask = self.PreMotion(pre_data, valid_id)  # reverses history
        fut_motion, fut_motion_mask = self.FutureMotion(fut_data, valid_id)

        data = {
                'pre_motion': pre_motion,
                'fut_motion': fut_motion,
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
                'frame_scale': 10,
                'frame': frame,
        }

        return data

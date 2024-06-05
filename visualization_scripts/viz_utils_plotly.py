import plotly.graph_objs as go
import numpy as np


COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

COCO_CONNECTIVITIES = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12],
                       [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]
H36M_FULL_CONNECTIVITIES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                            (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                            (21, 23)]
H36M_CONNECTIVITIES = [(0, 1), (1, 2), (2, 3), (0, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (10, 13),
                       (13, 14), (14, 15), (10, 16), (16, 17), (17, 18)]
BLAZEPOSE_CONNECTIVITIES = [(1, 2), (1, 5), (2, 3), (3, 7), (5, 6), (6, 7), (7, 9), (6, 8), (8, 10), (5, 4), (4, 11),
                            (11, 13), (13, 15), (15, 17), (17, 19), (19, 21), (6, 12), (12, 14), (14, 16), (16, 18),
                            (18, 20), (20, 22), (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (27, 29),
                            (29, 31), (26, 28), (28, 30), (30, 32)]

class AnimObjPose3d:
    def __init__(self):
        self.update = None
        self.fig = go.Figure()

    def plot_traj_anim(self, gt_history, gt_future, positions, bounds=None, ax=None):
        """
        Create a 3D video of the pose.
        gt_history: shape (num_peds, num_kp=33, obs_len, 3)
        positions: shape (num_peds, obs_len, 3)
        """
        obs_len = gt_history.shape[2]

        if positions is not None:
            # Add additional dimension for z-axis
            positions = np.concatenate([positions, np.zeros((*positions.shape[:-1], 1))], axis=-1).transpose(1, 0, 2)[:, None]
            gt_history = gt_history * 1 + positions[:, :, :obs_len]
            gt_future = gt_future * 1 + positions[:, :, obs_len:]

        stuff = np.concatenate([gt_history, gt_future], axis=2).reshape(-1, 3)

        self.ax_set_up(self.fig, stuff, bounds, True)

        def update(frame_idx):
            """
            Update the plot for a given frame.
            """
            nonlocal obs_len
            self.fig.data = []  # Clear the figure to draw a new pose
            self.ax_set_up(self.fig, stuff, bounds, True)
            if frame_idx < obs_len:
                self.draw_pose_3d_single_frame(gt_history, self.fig, True, frame_idx=frame_idx)
            else:
                self.draw_pose_3d_single_frame(gt_future, self.fig, False, frame_idx=frame_idx-obs_len)

        self.update = update

    def ax_set_up(self, fig, stuff=None, bounds=None, invert_yaxis=False):
        """ stuff is (N_examples, 2 or 3) """
        layout = {}
        if invert_yaxis:
            layout['scene'] = {'yaxis': {'autorange': 'reversed'}}

        if bounds is None:
            assert stuff is not None
            center = (stuff.min(axis=0) + stuff.max(axis=0)) / 2
            width_xyz = (stuff.max(axis=0) - stuff.min(axis=0))
            width = width_xyz.max()

            dim_min_x = center[0] - width / 2
            dim_max_x = center[0] + width / 2
            dim_min_y = center[1] - width / 2
            dim_max_y = center[1] + width / 2

            layout['scene'] = {
                'xaxis': {'range': [dim_min_x, dim_max_x]},
                'yaxis': {'range': [dim_min_y, dim_max_y]}
            }

            if stuff.shape[1] == 3:
                dim_min_z = center[2] - width / 2
                dim_max_z = center[2] + width / 2
                layout['scene']['zaxis'] = {'range': [dim_min_z, dim_max_z]}

        else:
            dim_min_x, dim_min_y, dim_max_x, dim_max_y = bounds[:4]
            layout['scene'] = {
                'xaxis': {'range': [dim_min_x, dim_max_x]},
                'yaxis': {'range': [dim_min_y, dim_max_y]}
            }
            if len(bounds) == 6:
                dim_min_z, dim_max_z = bounds[-2], bounds[-1]
                layout['scene']['zaxis'] = {'range': [dim_min_z, dim_max_z]}

        fig.update_layout(layout)

    def draw_pose_3d_single_frame(self, data, fig, is_history, frame_idx):
        colors = COLORS[:data.shape[0]]
        for i, color in enumerate(colors):
            pose = data[i, :, frame_idx]
            fig.add_trace(go.Scatter3d(
                x=pose[:, 0],
                y=pose[:, 1],
                z=pose[:, 2],
                mode='markers+lines',
                marker=dict(color=color),
                line=dict(color=color)
            ))

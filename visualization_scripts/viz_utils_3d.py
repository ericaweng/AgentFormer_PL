import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation


CONNECTIVITY_DICT = [
    (0, 1),
    (1, 2),
    (2, 3),
    (0, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (10, 13),
    (13, 14),
    (14, 15),
    (10, 16),
    (16, 17),
    (17, 18),
]


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


def draw_pose_3d_single_frame(pose, ax, color, frame_idx):
    """
    Draw a single pose for a given frame index.
    """
    vals = pose[0, :, frame_idx]
    for j1, j2 in CONNECTIVITY_DICT:
        x = np.array([vals[j1, 0], vals[j2, 0]])
        y = np.array([vals[j1, 1], vals[j2, 1]])
        z = np.array([vals[j1, 2], vals[j2, 2]])
        ax.plot(x, y, z, lw=2, color=color)


def ax_set_up(ax, stuff):
    if stuff is not None:
        center = (stuff.min(axis=(0, 1, 2)) + stuff.max(axis=(0, 1, 2))) / 2
        width_xyz = (stuff.max(axis=(0, 1, 2)) - stuff.min(axis=(0, 1, 2)))
        width = width_xyz.max()

        dim_min_x = center[0] - width / 2
        dim_max_x = center[0] + width / 2
        dim_min_y = center[1] - width / 2
        dim_max_y = center[1] + width / 2
        dim_min_z = center[2] - width / 2
        dim_max_z = center[2] + width / 2
        ax.set_xlim(dim_min_x, dim_max_x)
        ax.set_ylim(dim_min_y, dim_max_y)
        ax.set_zlim(dim_min_z, dim_max_z)
        # ax.view_init(azim=0, elev=45)
    else:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(0, 2.0)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")


class AnimObjPose:
    def __init__(self):
        self.update = None
    def video_pose_3d(self, pred_future, gt_history, gt_future, ax=None, bounds=None, filename=None, fps=25):
        """
        Create a 3D video of the pose.
        """
        obs_len = gt_history.shape[2]
        stuff = np.concatenate([pred_future, gt_history, gt_future], axis=2)
        if bounds is None:
            ax_set_up(ax, stuff)
        else:
            dim_min_x, dim_min_y, dim_min_z, dim_max_x, dim_max_y, dim_max_z = bounds
            ax.set_xlim(dim_min_x, dim_max_x)
            ax.set_ylim(dim_min_y, dim_max_y)
            ax.set_zlim(dim_min_z, dim_max_z)

        def update(frame_idx):
            """
            Update the plot for a given frame.
            """
            nonlocal obs_len
            ax.cla()  # Clear the axis to draw a new pose
            ax_set_up(ax, stuff)
            if frame_idx < obs_len:
                draw_pose_3d_single_frame(gt_history, ax, color='g', frame_idx=frame_idx)
            else:
                draw_pose_3d_single_frame(pred_future, ax, color='r', frame_idx=frame_idx-obs_len)
                draw_pose_3d_single_frame(gt_future, ax, color='g', frame_idx=frame_idx-obs_len)

        self.update = update

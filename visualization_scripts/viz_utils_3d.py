import tempfile
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualization_scripts.viz_utils2 import AnimObj


CONNECTIVITY_DICT = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                     (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                     (21, 23)]
CONNECTIVITY_DICT_H36M = [
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
    for i in range(pose.shape[0]):  # for each ped
        vals = pose[i, :, frame_idx]
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

def plot_anim_grid_3d(save_fn=None, title=None, plot_size=None, list_of_arg_dicts=None):
    """
    AO: the animation object to use. different AOs plot different things, and take different arguments.
        also, AO can be a list of different Animatin objects to use.
        the AO object should have a function called plot_traj_anim, which takes in at the very least,
        two args called ax and bounds
    list_of_arg_dicts: a list of dicts, where each dict is a set of arguments to pass to AO.plot_traj_anim
    """
    # set up figure
    if plot_size is None:
        if len(list_of_arg_dicts) > 4:
            num_plots_height = 2
            num_plots_width = int((len(list_of_arg_dicts) + 1) / 2)
        else:
            num_plots_height = 1
            num_plots_width = len(list_of_arg_dicts)
    else:
        num_plots_width, num_plots_height = plot_size

    assert num_plots_width * num_plots_height >= len(list_of_arg_dicts), \
        f'plot_size ({plot_size}) must be able to accomodate {len(list_of_arg_dicts)} graphs'

    fig = plt.figure(figsize=(7.5 * num_plots_width, 5 * num_plots_height))
    ax3d = fig.add_subplot(num_plots_height, num_plots_width, 1, projection='3d')
    axes = []
    for i in range(2, num_plots_width * num_plots_height + 1):
        axes.append(fig.add_subplot(num_plots_height, num_plots_width, i))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    # observation steps and prediction steps
    obs_len = 8
    pred_len = 12

    # 3d bounds for first plot
    bounds_3d = []
    for graph in list_of_arg_dicts:
        for key, val in graph.items():
            if "history" in key or 'future' in key:
                if len(bounds_3d) > 0:
                    assert val.shape[-1] == bounds_3d[-1].shape[-1], \
                        f"all trajectories must have same number of dimensions ({val.shape[-1]} != {bounds_3d[-1].shape[-1]})"
                bounds_3d.append(np.array(val).reshape(-1, val.shape[-1]))
    bounds_3d = np.concatenate(bounds_3d)
    bounds_3d = [*(np.min(bounds_3d, axis=0) - 0.2), *(np.max(bounds_3d, axis=0) + 0.2)]

    anim_graphs = []
    ao = AnimObjPose()
    anim_graphs.append(ao)
    ao.plot_traj_anim(**list_of_arg_dicts[0], ax=ax3d, bounds=bounds_3d)

    # set global plotting bounds (same for each sample)
    bounds = []
    for graph in list_of_arg_dicts:
        for key, val in graph.items():
            if 'traj' in key:
                if len(bounds) > 0:
                    assert val.shape[-1] == bounds[-1].shape[-1], \
                        f"all trajectories must have same number of dimensions ({val.shape[-1]} != {bounds[-1].shape[-1]})"
                bounds.append(np.array(val).reshape(-1, val.shape[-1]))
    bounds = np.concatenate(bounds)
    bounds = [*(np.min(bounds, axis=0) - 0.2), *(np.max(bounds, axis=0) + 0.2)]

    # instantiate animation object for each graph
    figs = []
    for ax_i, (arg_dict, ax) in enumerate(zip(list_of_arg_dicts[1:], axes)):
        ao = AnimObj()
        anim_graphs.append(ao)
        ao.plot_traj_anim(**arg_dict, ax=ax, bounds=bounds)

    def mass_update(frame_i):
        nonlocal figs, fig
        for ag in anim_graphs:
            ag.update(frame_i)
        figs.append(fig_to_array(fig))

    anim = animation.FuncAnimation(fig, mass_update, frames=obs_len + pred_len, interval=500)
    # save animation
    if save_fn is not None:
        anim.save(save_fn)
        print(f"saved animation to {save_fn}")
        plt.close(fig)
    else:
        with tempfile.TemporaryDirectory() as output_dir:
            anim.save(f"{output_dir}/temp.gif")
    return figs


class AnimObjPose:
    def __init__(self):
        self.update = None
    def plot_traj_anim(self, gt_history, gt_future, ax=None, bounds=None):
        """
        Create a 3D video of the pose.
        """
        obs_len = gt_history.shape[2]
        stuff = np.concatenate([gt_history, gt_future], axis=2)
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
                draw_pose_3d_single_frame(gt_future, ax, color='b', frame_idx=frame_idx-obs_len)

        self.update = update

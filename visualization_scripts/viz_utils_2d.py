import tempfile
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visualization_scripts.viz_utils2 import AnimObj


COCO_CONNECTIVITIES_LIST = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12],
                                         [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image

def ax_set_up_2d(ax, stuff):
    if stuff is not None:
        center = (stuff.min(axis=(0, 1, 2)) + stuff.max(axis=(0, 1, 2))) / 2
        width_xyz = (stuff.max(axis=(0, 1, 2)) - stuff.min(axis=(0, 1, 2)))
        width = width_xyz.max()

        dim_min_x = center[0] - width / 2
        dim_max_x = center[0] + width / 2
        dim_min_y = center[1] - width / 2
        dim_max_y = center[1] + width / 2
        ax.set_xlim(dim_min_x, dim_max_x)
        ax.set_ylim(dim_min_y, dim_max_y)
    else:
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")


def plot_anim_grid_2d(save_fn=None, title=None, plot_size=None, list_of_arg_dicts=None):
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
    ax2d = fig.add_subplot(num_plots_height, num_plots_width, 1)
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

    # 3d bounds for first pt
    bounds_2d = []
    for graph in list_of_arg_dicts:
        for key, val in graph.items():
            if "history" in key or 'future' in key:
                if len(bounds_2d) > 0:
                    assert val.shape[-1] == bounds_2d[-1].shape[-1], \
                        f"all trajectories must have same number of dimensions ({val.shape[-1]} != {bounds_2d[-1].shape[-1]})"
                bounds_2d.append(np.array(val).reshape(-1, val.shape[-1]))
    bounds_2d = np.concatenate(bounds_2d)
    bounds_2d = [*(np.min(bounds_2d, axis=0) - 0.2), *(np.max(bounds_2d, axis=0) + 0.2)]

    anim_graphs = []
    ao = AnimObjPose2d()
    anim_graphs.append(ao)
    ao.plot_traj_anim(**list_of_arg_dicts[0], ax=ax2d, bounds=bounds_2d)

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


class AnimObjPose2d:
    def __init__(self):
        self.update = None
    def plot_traj_anim(self, gt_history, gt_future, positions=None,ax=None, bounds=None):
        """
        Create a 2D video of the pose.
        """
        obs_len = gt_history.shape[2]

        if positions is not None:
            gt_history[...,1] = - gt_history[...,1]
            gt_history = gt_history * 1 + positions[:obs_len, :, :].transpose(1, 0, 2)[:,None]
            gt_future[...,1] = - gt_future[...,1]
            gt_future = gt_future * 1 + positions[obs_len:, :, :].transpose(1, 0, 2)[:,None]

        stuff = np.concatenate([gt_history, gt_future], axis=2)

        if bounds is None:
            ax_set_up_2d(ax, stuff)
        else:
            dim_min_x, dim_min_y, dim_max_x, dim_max_y = bounds
            ax.set_xlim(dim_min_x, dim_max_x)
            ax.set_ylim(dim_min_y, dim_max_y)

        def update(frame_idx):
            """
            Update the plot for a given frame.
            """
            nonlocal obs_len
            ax.cla()  # Clear the axis to draw a new pose
            ax_set_up_2d(ax, stuff)
            if frame_idx < obs_len:
                draw_pose_2d_single_frame(gt_history, ax, thin=True, frame_idx=frame_idx)
            else:
                draw_pose_2d_single_frame(gt_future, ax, thin=False, frame_idx=frame_idx - obs_len)

        self.update = update


COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
def draw_pose_2d_single_frame(pose, ax, thin, frame_idx):
    """
    Draw a single pose for a given frame index.
    """
    for i in range(pose.shape[0]):  # for each ped
        vals = pose[i, :, frame_idx]
        for j1, j2 in COCO_CONNECTIVITIES_LIST:
            x = np.array([vals[j1, 0], vals[j2, 0]])
            y = np.array([vals[j1, 1], vals[j2, 1]])
            ax.plot(x, y, lw=2 if thin else 4, color=COLORS[i % len(COLORS)])

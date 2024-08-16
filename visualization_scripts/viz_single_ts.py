import numpy as np

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from mpl_toolkits.mplot3d import Axes3D
from jrdb_toolkit.visualisation.visualize_constants import BLAZEPOSE_CONNECTIVITIES, OPENPOSE44_CONNECTIONS


def plot_scene(obs_traj, gt_traj=None, pred_traj=None, ped_ids=None, ped_radius=0.1, frame_id=None, ped_colors=None,
               bounds=None, plot_ped_texts=False, heading=None, save_fn=None):
    """obs_traj: (obs_len, num_peds, 2)
    gt_traj: (pred_len, num_peds, 2)
    pred_traj: (pred_len, num_peds, 2)
    """

    obs_len, num_peds, _ = obs_traj.shape
    if pred_traj is not None:
        pred_len, _, _ = pred_traj.shape

    if ped_colors is None:
        ped_colors = {ped_id: np.array(color) for color, ped_id in zip(plt.cm.tab20.colors, ped_ids)}

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # automatically calculate bounds based on obs_traj, gt_traj, and pred_traj
    if bounds is None:
        x_min = np.min(obs_traj[..., 0])
        x_max = np.max(obs_traj[..., 0])
        y_min = np.min(obs_traj[..., 1])
        y_max = np.max(obs_traj[..., 1])
        if gt_traj is not None:
            x_max = max(x_max, np.max(gt_traj[..., 0]))
            x_min = min(x_min, np.min(gt_traj[..., 0]))
            y_max = max(y_max, np.max(gt_traj[..., 1]))
            y_min = min(y_min, np.min(gt_traj[..., 1]))
        if pred_traj is not None:
            x_max = max(x_max, np.max(pred_traj[..., 0]))
            x_min = min(x_min, np.min(pred_traj[..., 0]))
            y_max = max(y_max, np.max(pred_traj[..., 1]))
            y_min = min(y_min, np.min(pred_traj[..., 1]))
        x_low, y_low, x_high, y_high = x_min - 1, y_min - 1, x_max + 1, y_max + 1
    else:
        x_low, y_low, x_high, y_high = bounds
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    ax.set_aspect("equal")

    # color and style properties
    text_offset_x = -0.5
    text_offset_y = -0.2
    lw = 4

    is_obs = frame_id < obs_len

    if is_obs:
        obs_alpha = 1
        gt_alpha = 0
        pred_alpha = 0
    else:
        obs_alpha = 0.5
        gt_alpha = 1
        pred_alpha = 1

    # plot obs traj
    for ped_i in range(num_peds):
        ped_id = ped_ids[ped_i]
        color = ped_colors[ped_id][[2, 1, 0]]
        ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=color,
                                                    alpha=obs_alpha, zorder=0))
        ax.add_artist(mlines.Line2D(*obs_traj[:frame_id+1, ped_i].T, color=color, alpha=obs_alpha,
                                    marker=None, linestyle='-', linewidth=lw, zorder=1))
    if not is_obs:
        # plot gt futures
        if gt_traj is not None:
            for ped_i in range(num_peds):
                ped_id = ped_ids[ped_i]
                color = ped_colors[ped_id][[2, 1, 0]]
                gt_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], gt_traj[:frame_id-obs_len+1, ped_i]])
                ax.add_artist(mlines.Line2D(*gt_plus_last_obs.T, color=color, alpha=gt_alpha,
                                            marker=None, linestyle='-', linewidth=lw, zorder=1))
            # plot ped texts
            if plot_ped_texts:
                ped_texts = []
                for ped_i in range(num_peds):
                    int_text = ax.text(gt_traj[-1, ped_i, 0] + text_offset_x,
                                       gt_traj[-1, ped_i, 1] - text_offset_y,
                                       f'A{ped_ids[ped_i]}', color='black', fontsize=14)
                    ped_texts.append(ax.add_artist(int_text))

        # plot pred futures
        if pred_traj is not None:
            for ped_i in range(num_peds):
                ped_id = ped_ids[ped_i]
                color = ped_colors[ped_id][[2, 1, 0]]
                pred_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], pred_traj[:frame_id-obs_len+1, ped_i]])
                ax.add_artist(mlines.Line2D(*pred_plus_last_obs.T, color=color, alpha=pred_alpha,
                                            marker=None, linestyle='--', linewidth=lw, zorder=1))

        # plot last obs arrows
        small_arrows = False
        if heading is not None:
            traj = obs_traj
            last_arrow_starts = traj[-1]
            last_arrow_lengths = 0.1 * (traj[-1] - traj[-2]).reshape(-1, 2)
            for ped_i, (arrow_start, arrow_length) in enumerate(zip(last_arrow_starts, last_arrow_lengths)):
                ax.add_artist(patches.FancyArrow(*arrow_start, *arrow_length,
                                                 overhang=3,
                                                 head_width=.5,
                                                 head_length=.2,
                                                 color=ped_colors[ped_ids[ped_i]][[2, 1, 0]],
                                                 zorder=0,
                                                 linewidth=0.4 if small_arrows else 2))
    # Remove any margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    if save_fn is not None:
        plt.savefig(save_fn)
    else:
        img = fig_to_array(fig)
        plt.close(fig)
        return img


def plot_scene_3d(pose, ped_ids, ped_colors, pose_type, bounds=None, robot_loc=None, robot_yaw=None,
                  frame_id=None, ped_ids_pred=None, obs_traj=None, gt_traj=None, pred_traj=None,
                  point_cloud=None, vectors_3d=None):
    """
    Create a 3D plot image of the pedestrian poses for current timestep.
    """

    fig, ax = plt.subplots(1, 1, figsize=(20, 20), subplot_kw={'projection': '3d'})

    if robot_loc is not None:
        # plot robot as a red sphere
        ax.scatter(*robot_loc, s=200, color='blue', alpha=0.5)
        # arrow pointing toward robot yaw
    if robot_yaw is not None:
        arrow_start = robot_loc
        arrow_length = 1 * np.array([np.cos(robot_yaw), np.sin(robot_yaw), 0])
        # use quiver
        ax.add_artist(ax.quiver(*arrow_start, *arrow_length, color='b', alpha=0.5))

    for ped_i in range(pose.shape[0]):  # for each ped
        ped_id = ped_ids[ped_i]
        vals = pose[ped_i]
        for j1, j2 in BLAZEPOSE_CONNECTIVITIES if pose_type.lower() == 'blazepose' else OPENPOSE44_CONNECTIONS:
            x = np.array([vals[j1, 0], vals[j2, 0]])
            y = np.array([vals[j1, 1], vals[j2, 1]])
            z = np.array([vals[j1, 2], vals[j2, 2]])
            ax.plot(x, y, z, lw=1, color=ped_colors[ped_id][[2, 1, 0]])

    # trajectories plotting
    if obs_traj is not None and pred_traj is not None and gt_traj is not None:
        if obs_traj is not None:
            obs_len, num_peds, _ = obs_traj.shape
        if pred_traj is not None:
            pred_len, num_peds, _ = pred_traj.shape

        # color and style properties
        text_offset_x = -0.5
        text_offset_y = -0.2
        plot_ped_texts = False
        lw = 4

        is_obs = frame_id < obs_len

        if is_obs:
            obs_alpha = 1
            gt_alpha = 0
            pred_alpha = 0
        else:
            obs_alpha = 0.5
            gt_alpha = 1
            pred_alpha = 1

        # plot obs traj
        for ped_i, ped_id in enumerate(ped_ids_pred):
            color = ped_colors[ped_id][[2, 1, 0]]

            ax.scatter(*np.concatenate([obs_traj, gt_traj])[frame_id, ped_i], s=100, color=color, alpha=obs_alpha, zorder=0)
            ax.plot(*obs_traj[:frame_id + 1, ped_i].T, color=color, alpha=obs_alpha, linestyle='-', linewidth=lw, zorder=1)

            # plot ped texts
            if plot_ped_texts:
                ped_texts = []
                for ped_i in range(num_peds):
                    int_text = ax.text(gt_traj[-1, ped_i, 0] + text_offset_x, gt_traj[-1, ped_i, 1] - text_offset_y,
                                       f'A{ped_id}', color=color, fontsize=14)
                    ped_texts.append(ax.add_artist(int_text))

            if is_obs:
                continue
            # plot gt futures
            if gt_traj is not None:
                gt_plus_last_obs = np.concatenate(
                        [obs_traj[-1:, ped_i], gt_traj[:frame_id - obs_len + 1, ped_i]])
                ax.plot(*gt_plus_last_obs.T, color=color, alpha=gt_alpha, linestyle='-', linewidth=lw, zorder=1)

            # plot pred futures
            if pred_traj is not None:
                pred_plus_last_obs = np.concatenate(
                        [obs_traj[-1:, ped_i], pred_traj[:frame_id - obs_len + 1, ped_i]])
                ax.plot(*pred_plus_last_obs.T, color=color, alpha=pred_alpha, linestyle='--', linewidth=lw, zorder=1)

    # plot 2d point cloud
    filtered_points = point_cloud
    filtered_points = filtered_points[filtered_points[:, 2] > -0.2]
    plt.scatter(filtered_points[:, 0], filtered_points[:, 1], s=1)

    # Plot the 3D vectors in red
    if vectors_3d is not None:
        for start_point, end_point in vectors_3d:
            norm_vector = (end_point - start_point) / np.linalg.norm(end_point - start_point) * 0.5  # normalize to length 0.5
            ax.quiver(start_point[0], start_point[1], start_point[2],
                      norm_vector[0], norm_vector[1], norm_vector[2], color='magenta')

    # Remove any margins
    _ax_set_up(ax, pose.reshape(-1, pose.shape[-1]), bounds, True)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

    img = fig_to_array(fig)
    plt.close(fig)
    return img



def get_bounds(*stuff):
    assert np.all([s.shape[-1]==stuff[0].shape[-1] for s in stuff])
    stuff = np.concatenate([s.reshape(-1, s.shape[-1]) for s in stuff])
    stuff = np.nan_to_num(stuff)
    center = (stuff.min(axis=0) + stuff.max(axis=0)) / 2
    width_xyz = (stuff.max(axis=0) - stuff.min(axis=0))
    width = width_xyz.max()

    dim_min_x = center[0] - width / 2
    dim_max_x = center[0] + width / 2
    dim_min_y = center[1] - width / 2
    dim_max_y = center[1] + width / 2

    if stuff.shape[1] == 3:
        dim_min_z = 0
        dim_max_z = width
        return dim_min_x, dim_max_x, dim_min_y, dim_max_y, dim_min_z, dim_max_z
    return dim_min_x, dim_max_x, dim_min_y, dim_max_y


def _ax_set_up(ax, stuff=None, bounds=None, invert_yaxis=False):
    """ stuff is (N_examples, 2 or 3) """

    if invert_yaxis:
        ax.invert_yaxis()  # to match the 2d bev trajectory plot better

    if bounds is None:
        assert stuff is not None
        res = get_bounds(*stuff)
        dim_min_x, dim_max_x, dim_min_y, dim_max_y = res[:4]
        ax.set_xlim(dim_min_x, dim_max_x)
        ax.set_ylim(dim_min_y, dim_max_y)

        if stuff.shape[1] == 3:
            dim_min_z, dim_max_z = res[-2:]
            ax.set_zlim(dim_min_z, dim_max_z)

    else:
        assert len(bounds) == 4 or len(bounds) == 6
        dim_min_x, dim_max_x, dim_min_y, dim_max_y = bounds[:4]
        ax.set_xlim(dim_min_x, dim_max_x)
        ax.set_ylim(dim_min_y, dim_max_y)
        if len(bounds) == 6:
            dim_min_z, dim_max_z = bounds[-2:]
            ax.set_zlim(dim_min_z, dim_max_z)


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


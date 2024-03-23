import tempfile
import numpy as np

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.animation as animation


COCO_CONNECTIVITIES = [[1, 2], [0, 4], [3, 4], [8, 10], [5, 7], [10, 13], [14, 16], [4, 5], [7, 12],
                       [4, 8], [3, 6], [13, 15], [11, 14], [6, 9], [8, 11]]
H36M_FULL_CONNECTIVITIES = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12),
                            (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21), (20, 22),
                            (21, 23)]
H36M_CONNECTIVITIES = [ (0, 1), (1, 2), (2, 3), (0, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (10, 13),
                        (13, 14), (14, 15), (10, 16), (16, 17), (17, 18), ]


def fig_to_array(fig):
    fig.canvas.draw()
    fig_image = np.array(fig.canvas.renderer._renderer)

    return fig_image


def ax_set_up(ax, stuff):
    """ stuff is (N_examples, 2 or 3) """
    center = (stuff.min(axis=0) + stuff.max(axis=0)) / 2
    width_xyz = (stuff.max(axis=0) - stuff.min(axis=0))
    width = width_xyz.max()

    dim_min_x = center[0] - width / 2
    dim_max_x = center[0] + width / 2
    dim_min_y = center[1] - width / 2
    dim_max_y = center[1] + width / 2
    ax.set_xlim(dim_min_x, dim_max_x)
    ax.set_ylim(dim_min_y, dim_max_y)

    if stuff.shape[1] == 3:
        dim_min_z = center[2] - width / 2
        dim_max_z = center[2] + width / 2
        ax.set_zlim(dim_min_z, dim_max_z)

def get_grid_size_from_num_grids(n):
    """ returns a tuple of (width, height) for a grid of n plots """
    if 20 < n <= 25:
        return (5, 5)
    elif 15 < n <= 20:
        return (4, 5)
    elif 12 < n <= 15:
        return (3, 5)
    elif 8 < n <= 12:
        return (3, 4)
    elif 6 < n <= 8:
        return (2, 4)
    elif 4 < n <= 6:
        return (2, 3)
    elif n == 4:
        return (2, 2)
    elif n == 3:
        return (1, 3)
    elif n == 2:
        return (1, 2)
    elif n == 1:
        return (1, 1)
    else:
        return "Input out of defined range"


def plot_anim_grid(save_fn=None, title=None, list_of_arg_dicts=None, list_of_plotting_objs=None,
                   plot_size=None, obs_len=None, pred_len=None):
    """
    AO: the animation object to use. different AOs plot different things, and take different arguments.
        also, AO can be a list of different Animatin objects to use.
        the AO object should have a function called plot_traj_anim, which takes in at the very least,
        two args called ax and bounds
    list_of_arg_dicts: a list of dicts, where each dict is a set of arguments to pass to AO.plot_traj_anim
    """
    assert len(list_of_arg_dicts) == len(list_of_plotting_objs), \
        f'list_of_arg_dicts ({len(list_of_arg_dicts)}) must be the same length as list_of_arg_functions ({len(list_of_plotting_objs)})'

    if obs_len is None:
        obs_len = list_of_arg_dicts[-1]['obs_traj'].shape[0]
    if pred_len is None:
        pred_len = list_of_arg_dicts[-1]['gt_traj'].shape[0]

    if plot_size is None:
        num_plots_height, num_plots_width = get_grid_size_from_num_grids(len(list_of_arg_dicts))
    else:
        num_plots_height, num_plots_width = plot_size
        assert num_plots_width * num_plots_height >= len(list_of_arg_dicts), \
            f'plot_size ({plot_size}) must be able to accomodate {len(list_of_arg_dicts)} graphs'

    # make plots grid
    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(7.5 * num_plots_width, 5 * num_plots_height))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]
    fig.subplots_adjust(hspace=0.25)
    if title is not None:
        fig.suptitle(title, fontsize=16)

    # set global plotting bounds (same for each sample)
    bounds2d = []
    for graph in list_of_arg_dicts:
        for key, val in graph.items():
            if 'traj' in key:
                if len(bounds2d) > 0:
                    assert val.shape[-1] == bounds2d[-1].shape[-1], \
                        f"all trajectories must have same number of dimensions ({val.shape[-1]} != {bounds2d[-1].shape[-1]})"
                bounds2d.append(np.array(val).reshape(-1, val.shape[-1]))
    bounds2d = np.concatenate(bounds2d)
    bounds2d = [*(np.min(bounds2d, axis=0) - 0.2), *(np.max(bounds2d, axis=0) + 0.2)]

    anim_objects = []
    animation_frames = []
    for ao_class, arg_dict, ax in zip(list_of_plotting_objs, list_of_arg_dicts, axes):
        ao = ao_class()
        anim_objects.append(ao)
        if isinstance(ao, AnimObjPose2d) or isinstance(ao, AnimObjBEVTraj2d):
            bounds = bounds2d[:4]
        else:
            bounds = None
        ao.plot_traj_anim(**arg_dict, ax=ax, bounds=bounds)

    def mass_update(frame_i):
        nonlocal animation_frames, fig
        for ag in anim_objects:
            ag.update(frame_i)
        animation_frames.append(fig_to_array(fig))

    anim = animation.FuncAnimation(fig, mass_update, frames=obs_len + pred_len, interval=500)

    # save animation
    if save_fn is not None:
        anim.save(save_fn)
        print(f"saved animation to {save_fn}")
        plt.close(fig)
    else:
        with tempfile.TemporaryDirectory() as output_dir:
            anim.save(f"{output_dir}/temp.gif")
    return animation_frames


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

        stuff = np.concatenate([gt_history, gt_future], axis=2).reshape(-1, 2)

        if bounds is None:
            ax_set_up(ax, stuff)
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
            ax_set_up(ax, stuff)
            if frame_idx < obs_len:
                draw_pose_2d_single_frame(gt_history, ax, thin=True, frame_idx=frame_idx)
            else:
                draw_pose_2d_single_frame(gt_future, ax, thin=False, frame_idx=frame_idx - obs_len)

        self.update = update


class AnimObjPose3d:
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


COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

def draw_pose_2d_single_frame(pose, ax, thin, frame_idx):
    """
    Draw a single pose for a given frame index.
    """
    for i in range(pose.shape[0]):  # for each ped
        vals = pose[i, :, frame_idx]
        for j1, j2 in COCO_CONNECTIVITIES:
            x = np.array([vals[j1, 0], vals[j2, 0]])
            y = np.array([vals[j1, 1], vals[j2, 1]])
            ax.plot(x, y, lw=1 if thin else 2, color=COLORS[i % len(COLORS)])


def draw_pose_3d_single_frame(pose, ax, color, frame_idx):
    """
    Draw a single pose for a given frame index.
    """
    for i in range(pose.shape[0]):  # for each ped
        vals = pose[i, :, frame_idx]
        for j1, j2 in COCO_CONNECTIVITIES:
            x = np.array([vals[j1, 0], vals[j2, 0]])
            y = np.array([vals[j1, 1], vals[j2, 1]])
            z = np.array([vals[j1, 2], vals[j2, 2]])
            ax.plot(x, y, z, lw=2, color=color)


class AnimObjBEVTraj2d:
    def __init__(self):
        self.update = None

    def plot_traj_anim(self, obs_traj=None, save_fn=None, ped_radius=0.2, ped_discomfort_dist=0.2, gt_traj=None,
                       pred_traj=None, ped_num_label_on='gt', show_ped_ids=False, show_ped_pos=False, bounds=None,
                       scene_stats=None, avg_heading=None, last_heading=None, collision_mats=None, extend_last_frame=3,
                       show_ped_stats=False, text_time=None, text_fixed=None, grid_values=None,
                       plot_collisions_all=False, plot_title=None, ax=None, pred_alpha=None):
        # TODO show_ped_pos does not do ped pos for obs steps
        """
        obs_traj: shape (8, num_peds, 2) observation input to model, first 8 timesteps of the scene
        save_fn: file name where to save animation
        ped_diameter: collision threshold -- pedestrian radius * 2
        pred_traj_fake: tensor of shape (8 or 12 pred timesteps, num_peds, 2)
                        or tensor of shape (num_samples, 8 or 12 pred timesteps, num_peds, 2)
                        or list of tensors of shape (8 or 12, num_peds, 2)  (where each item are the samples predicted by a different model)
                        or list of tensors of shape (num_samples, 8 or 12 pred timesteps, num_peds, 2)
        show_ped_pos: whether to show the position of each ped next to the ped circle
        bounds: x_low, y_low, x_high, y_high: plotting bounds
                if not specified the min and max bounds of whichever trajectories are present are used
        pred_traj_gt: shape (8 or 12, num_peds, 2) ground-truth trajectory
        interaction_matrix: shape (num_peds, num_peds - 1) specifies which pairs belong to the given int_type.
                            only used for int_types that are pairwise, i.e. "linear" "static" etc. are not relevant.
                            np.sum(interaction_matrix, axis=-1) produces an "interaction level" for each ped, which
                            is used to color it in the plot. the more green a ped is, the greater number of peds it
                            shares that int_type with. the more blue, the fewer.
        int_type_abbv: used for the title and coloring peds
        scene_stats: statistics for each ped in the scene to plot
        collision mats: if already computed, plots when a collision occurs
        cmap_name: which color map to use for coloring pedestrians
        extend_last_frame: how many timesteps to extend the last frame so the viewer can better observe the full trajectory
        scatter_dots: a dict mapping labels to sets of np.ndarray scatter points, or a list of np.ndarray scatter points,
                     or an np.ndarray set of scatter points, to plot
        show_ped_stats: (bool) whether to display statistics for each pedestrian on the plot
        text_time (list): list of strings of len = num_timesteps, of text to plot that changes each timestep
        grid_values (np.array): colored grid to plot, for debugging purposes
        plot_collisions_all: if True, and collision_mats is specified, plots obs step and pred step collisions
                             o/w: plots only pred step fake collisions
        """
        assert not all([obs_traj is None, pred_traj is None, gt_traj is None]), "at least one of obs_traj, pred_traj_fake, or pred_traj_gt must be supplied"

        # instantiate ax if not exist
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        else:
            fig = None
        plot_title = f"{plot_title}\n" if plot_title is not None else ""
        ax.set_title(plot_title, fontsize=16)
        ax.set_aspect("equal")

        # obs len
        if obs_traj is not None:
            obs_len = obs_traj.shape[0]
        else:
            obs_len = 0
        # pred len
        if gt_traj is not None:
            pred_len = gt_traj.shape[0]
        elif pred_traj is not None:
            pred_len = pred_traj[0].shape[1]
        else:
            pred_len = 0
        # num_peds
        if obs_traj is not None:
            num_peds = obs_traj.shape[1]
        elif gt_traj is not None:
            num_peds = gt_traj.shape[1]
        elif pred_traj is not None:
            num_peds = pred_traj[0].shape[2]
        else:
            raise RuntimeError

        # calculate bounds automatically
        if bounds is None:
            all_traj = np.zeros((0, 2))
            if obs_traj is not None:
                all_traj = obs_traj.reshape(-1, 2)
            if gt_traj is not None:
                all_traj = np.concatenate([all_traj, gt_traj.reshape(-1, 2)])
            if pred_traj is not None:
                all_traj = np.concatenate([all_traj, *[p.reshape(-1, 2) for ptf in pred_traj for p in ptf]])
            x_low, x_high = np.min(all_traj[:, 0]) - ped_radius, np.max(all_traj[:, 0]) + ped_radius
            y_low, y_high = np.min(all_traj[:, 1]) - ped_radius, np.max(all_traj[:, 1]) + ped_radius
        else:  # set bounds as specified
            x_low, y_low, x_high, y_high = bounds
        ax.set_xlim(x_low, x_high)
        ax.set_ylim(y_low, y_high)

        # color and style properties
        text_offset_x = 0.2
        text_offset_y = 0.2
        obs_alpha = 1  # how much alpha to plot obs traj
        if pred_alpha is None:
            pred_alpha = 0.5  # how much alpha to plot gt traj, if they exist

        # each ped a different color
        cmap_fake = lambda i:  COLORS[i%len(COLORS)]
        cmap_real = lambda i:  COLORS[i%len(COLORS)]

        # add scene-related stats as descriptive text
        if show_ped_stats:
            if scene_stats is not None:
                values = map(lambda x: f"{x:0.2f}", scene_stats.values())
                scene_stats_text = f'{" / ".join(map(str, scene_stats.keys()))}\n{" / ".join(values)}'
                ax.add_artist(plt.text(x_low + 0.1, y_high + .2, scene_stats_text, fontsize=8))
                # ax.add_artist(plt.text(x_low + 0.1, y_high - .3, 'obs // pred (avg_speed / std_speed / smoothness)', fontsize=8))

        # ## text that changes each frame
        if text_time is not None:
            text_over_time = ax.text(14, 6, "", fontsize=10, color='k', weight='bold')
            ax.add_artist(text_over_time)

        ## text that stays fixed each frame
        offset_lower = 0.1
        text_fixed_fs = 16
        if isinstance(text_fixed, str):
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text_fixed, fontsize=text_fixed_fs))
        elif isinstance(text_fixed, list):
            text = "\n".join(text_fixed)
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=text_fixed_fs))
        elif isinstance(text_fixed, dict):
            text = "\n".join([f'{k}: {v:0.3f}' for k, v in text_fixed.items()])
            ax.add_artist(ax.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=text_fixed_fs))
        else:
            if text_fixed is not None:
                raise NotImplementedError("text_fixed is unrecognized format")

        # ped graph elements
        circles_gt, circles_fake, last_obs_circles, lines_pred_gt, lines_obs_gt, lines_pred_fake = [], [], [], [], [], []

        # plot circles to represent peds
        last_heading_arrows = []
        avg_heading_arrows = []

        for ped_i in range(num_peds):
            color_real = cmap_real(ped_i % num_peds)
            color_fake = cmap_fake(ped_i % num_peds)

            # plot ground-truth obs and pred
            if obs_traj is not None:
                circles_gt.append(ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=color_real, zorder=0)))
                line_obs_gt = mlines.Line2D(*obs_traj[0:1].T, color=color_real, marker=None, linestyle='-', linewidth=5,
                                            alpha=obs_alpha, zorder=0)
                lines_obs_gt.append(ax.add_artist(line_obs_gt))

                # Plot body heading direction
                if last_heading is not None:
                    last_obs_circles.append(ax.add_artist(plt.Circle(obs_traj[-1, ped_i], ped_radius, fill=True,
                                                                     alpha=0.3, color=color_real, zorder=10,
                                                                     visible=False)))
                    last_heading_arrows.append(ax.arrow(*obs_traj[-1, ped_i], *last_heading[ped_i], head_width=0.05,
                                                        head_length=0.1, fc='r', ec='r', visible=False, zorder=15))
                if avg_heading is not None:
                    avg_heading_arrows.append(ax.arrow(*obs_traj[-1, ped_i], *avg_heading[ped_i], head_width=0.05,
                                                        head_length=0.1, fc='b', ec='b', visible=False, zorder=15))

                # # Plot head heading direction (different color)
                # # Replace head_heading with your actual head heading data
                # head_heading = head_headings[sorted_frame_ids[frame_id]][ped_id][0]
                # ax.arrow(pos[0], pos[1], head_heading[0], head_heading[1], head_width=0.05, head_length=0.1, fc='g',
                #          ec='g')

            if gt_traj is not None:
                if obs_traj is None:
                    circles_gt.append(ax.add_artist(plt.Circle(gt_traj[0, ped_i], ped_radius, fill=True, color=color_real, zorder=0)))
                line_pred_gt = mlines.Line2D(*gt_traj[0:1].T, color=color_real, marker=None, linestyle='-', linewidth=5,
                                             alpha=pred_alpha, zorder=0, visible=False)
                lines_pred_gt.append(ax.add_artist(line_pred_gt))

            if pred_traj is not None:  # plot fake pred trajs
                circle_fake = plt.Circle(pred_traj[0, ped_i], ped_radius, fill=True,
                                         color=color_fake,
                                         alpha=obs_alpha, visible=False, zorder=1)
                circles_fake.append(ax.add_artist(circle_fake))

                line_pred_fake = mlines.Line2D(*pred_traj[0:1].T, color=color_fake,
                                               linestyle='--',
                                               alpha=obs_alpha, zorder=2, linewidth=5,
                                               visible=False)

                lines_pred_fake.append(ax.add_artist(line_pred_fake))

        # label peds with id
        ped_texts = []
        if show_ped_ids:
            if ped_num_label_on == 'gt':
                circles_to_plot_ped_num = circles_gt
            elif ped_num_label_on == 'pred' or obs_traj is None and gt_traj is None:
                circles_to_plot_ped_num = circles_fake
            else:
                raise RuntimeError
            for ped_i, circle in enumerate(circles_to_plot_ped_num):
                int_text = ax.text(circle.center[0] + text_offset_x, circle.center[1] - text_offset_y,
                                   str(ped_i), color='black', fontsize=8)
                ped_texts.append(ax.add_artist(int_text))

        if show_ped_pos:
            ped_pos_texts_obs = []
            for ped_i, circle in enumerate(circles_gt):
                ped_pos_text = f"{circle.center[0]:0.1f}, {circle.center[1]:0.1f}"
                ped_pos_texts_obs.append(ax.add_artist(ax.text(circle.center[0] + text_offset_x, circle.center[1] + text_offset_y,
                                                               ped_pos_text, fontsize=8,)))
            ped_pos_texts = []
            for ped_i, circle in enumerate(circles_fake):
                ped_pos_text = f"{circle.center[0]:0.1f}, {circle.center[1]:0.1f}"
                ped_pos_texts.append(ax.add_artist(ax.text(circle.center[0] + text_offset_x, circle.center[1] + text_offset_y,
                                                   ped_pos_text, fontsize=8, visible=False)))

        # plot collision circles for predictions only
        if collision_mats is not None:
            collide_circle_rad = (ped_radius + ped_discomfort_dist)
            # assert collision_mats.shape == (pred_len, num_peds, num_peds)
            collision_circles = [ax.add_artist(plt.Circle((0, 0), collide_circle_rad, fill=False, zorder=50, visible=False))
                                 for _ in range(num_peds)]
            collision_texts = [ax.add_artist(ax.text(0, 0, "", visible=False, fontsize=8)) for _ in range(num_peds)]
            collision_delay = 3
            yellow = (.9, .5, 0, .4)
            collided_delays = np.zeros(num_peds)

        ax.tick_params(
                axis='both',  # changes apply to both x and y-axis
                which='both',  # both major and minor ticks are affected
                bottom=True,#False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                left=True,#False,  # ticks along the left edge are off
                right=False,  # ticks along the right edge are off
                labelbottom=True,#False,  # labels along the bottom edge are off
                labelleft=True,#False  # labels along the left edge are off
        )

        # heatmap
        if grid_values is not None:
            x, y = np.meshgrid(np.linspace(*bounds[:2], grid_values.shape[1] + 1),
                               np.linspace(*bounds[2:4], grid_values.shape[2] + 1))
            z = grid_values[0]

            z_min, z_max = np.min(np.array(z)), np.max(np.array(z))
            state_mesh = ax.pcolormesh(x, y, z, alpha=.8, vmin=0, vmax=1, zorder=3)

        ## animation update function
        def update(frame_i):
            nonlocal x, y
            # for replicating last scene
            if frame_i >= obs_len + pred_len:
                return

            # heatmap
            if grid_values is not None and frame_i < obs_len + pred_len - 1:
                nonlocal state_mesh, x, y
                z = grid_values[frame_i]
                normed_z = ((z - z_min) / (z_max - z_min)).reshape(x.shape[0] - 1, x.shape[1] - 1)
                state_mesh.remove()
                state_mesh = ax.pcolormesh(x, y, normed_z, alpha=.1, vmin=0, vmax=1, zorder=1)

            # move the real and pred (fake) agent
            if frame_i < obs_len:
                for ped_i, (circle_gt, line_obs_gt) in enumerate(zip(circles_gt, lines_obs_gt)):
                    circle_gt.center = obs_traj[frame_i, ped_i]
                    line_obs_gt.set_data(*obs_traj[0:frame_i + 1, ped_i].T)
                    if show_ped_pos and len(ped_pos_texts_obs) > 0:
                        ped_pos_text = f"{circle_gt.center[0]:0.1f}, {circle_gt.center[1]:0.1f}"
                        ped_pos_texts_obs[ped_i].set_text(ped_pos_text)
                        ped_pos_texts_obs[ped_i].set_position((circle_gt.center[0] + text_offset_x, circle_gt.center[1] - text_offset_y))
                for ped_i, circle_fake in enumerate(circles_fake):
                    circle_fake.center = obs_traj[frame_i, ped_i]
                if show_ped_pos:
                    [text.set_visible(True) for text in ped_pos_texts_obs]
                    [text.set_visible(False) for text in ped_pos_texts]

                # move the pedestrian texts (ped number and relation)
                for ped_text, circle in zip(ped_texts, circles_gt):  # circles_to_plot_ped_num):
                    ped_text.set_position((circle.center[0] + text_offset_x, circle.center[1] - text_offset_y))

                # set last heading vector and obs circles
                if frame_i == obs_len - 1:
                    if last_heading is not None:
                        for last_heading_arrow in last_heading_arrows:
                            last_heading_arrow.set_visible(True)
                        if avg_heading is not None:
                            for avg_heading_arrow in avg_heading_arrows:
                                avg_heading_arrow.set_visible(True)
                    for last_obs_circ in last_obs_circles:
                        last_obs_circ.set_visible(True)

            elif frame_i == obs_len:
                [cf.set_visible(True) for cf in circles_fake]
                if show_ped_pos:
                    [text.set_visible(True) for text in ped_pos_texts]
                    [text.set_visible(False) for text in ped_pos_texts_obs]
                for circle_gt in circles_gt:
                    circle_gt.set_radius(ped_radius * 0.5)
                    circle_gt.set_alpha(0.3)
                for line_obs_gt in lines_obs_gt:
                    line_obs_gt.set_alpha(0.2)
                if gt_traj is not None:
                    for line_pred_gt in lines_pred_gt:
                        line_pred_gt.set_visible(True)
                if pred_traj is not None:
                    for line_pred_fake in lines_pred_fake:
                        line_pred_fake.set_visible(True)

                for last_obs_circ in last_obs_circles:
                    last_obs_circ.set_radius(ped_radius * 0.75)
                    last_obs_circ.set_alpha(0.3)

            if obs_len <= frame_i < obs_len + pred_len:
                if gt_traj is not None:
                    # assert len(circles_gt) == len(lines_pred_gt) == len(ped_texts), f'{len(circles_gt)}, {len(lines_pred_gt)}, {len(ped_texts)} should all be equal'
                    for ped_i, (circle_gt, line_pred_gt) in enumerate(zip(circles_gt, lines_pred_gt)):
                        circle_gt.center = gt_traj[frame_i - obs_len, ped_i]
                        if obs_traj is not None:
                            last_obs_pred_gt = np.concatenate([obs_traj[-1:, ped_i], gt_traj[0:frame_i + 1 - obs_len, ped_i]])
                        else:
                            last_obs_pred_gt = gt_traj[0:frame_i + 1 - obs_len, ped_i]
                        line_pred_gt.set_data(*last_obs_pred_gt.T)
                        # move the pedestrian texts (ped number and relation)
                        if len(ped_texts) > 0:
                            ped_texts[ped_i].set_position((circle_gt.center[0] + text_offset_x, circle_gt.center[1] - text_offset_y))

                if pred_traj is not None:
                    assert len(lines_pred_fake) == len(circles_fake)
                    for ped_i, (circle_fake, line_pred_fake) in enumerate(zip(circles_fake, lines_pred_fake)):
                        circle_fake.center = pred_traj[frame_i - obs_len, ped_i]
                        if obs_traj is not None:
                            last_obs_pred_fake = np.concatenate([obs_traj[-1:, ped_i], pred_traj[0:frame_i + 1 - obs_len, ped_i]])
                        else:
                            last_obs_pred_fake = pred_traj[0:frame_i + 1 - obs_len, ped_i]
                        line_pred_fake.set_data(*last_obs_pred_fake.T)
                        if show_ped_pos and len(ped_pos_texts) > 0:
                            ped_pos_text = f"{circle_fake.center[0]:0.1f}, {circle_fake.center[1]:0.1f}"
                            ped_pos_texts[ped_i].set_text(ped_pos_text)
                            ped_pos_texts[ped_i].set_position((circle_fake.center[0] + text_offset_x, circle_fake.center[1] - text_offset_y))

            # update collision circles (only if we are during pred timesteps)
            if (plot_collisions_all or obs_len <= frame_i <= obs_len + pred_len) and collision_mats is not None:
                assert len(collision_mats.shape) == 3 and collision_mats.shape[1] == collision_mats.shape[2], 'collision mats is not square'
                if plot_collisions_all:
                    assert len(collision_mats) == obs_len + pred_len, f'plot_collisons_all is {plot_collisions_all}, so collision_mat size should be {obs_len + pred_len} but is {len(collision_mats)}'
                else:
                    assert len(collision_mats) == pred_len, f'plot_collisons_all is {plot_collisions_all}, so collision_mat size should be {pred_len} but is {len(collision_mats)}'

                if pred_traj is not None and obs_traj is not None:
                    obs_gt_fake = np.concatenate([obs_traj, pred_traj])
                elif gt_traj is not None and obs_traj is not None:
                    obs_gt_fake = np.concatenate([obs_traj, gt_traj])
                elif pred_traj is not None:
                    obs_gt_fake = pred_traj
                elif gt_traj is not None:
                    obs_gt_fake = gt_traj
                else:
                    raise RuntimeError

                for ped_i in range(num_peds):
                    # new frame; decrease the text disappearance delay by 1
                    if collided_delays[ped_i] > 0:
                        collided_delays[ped_i] -= 1
                    for ped_j in range(ped_i):
                        if plot_collisions_all:
                            collision_frame_idx = frame_i
                        else:
                            collision_frame_idx = frame_i - obs_len
                        if collided_delays[ped_i] > 0:  # still in delay, circle doesn't disappear
                            break
                        elif collision_mats[collision_frame_idx, ped_i, ped_j]:
                            ## put the center of the circle in the point between the two ped centers
                            x = (obs_gt_fake[frame_i][ped_i][0] + obs_gt_fake[frame_i][ped_j][0]) / 2
                            y = (obs_gt_fake[frame_i][ped_i][1] + obs_gt_fake[frame_i][ped_j][1]) / 2
                            collision_circles[ped_i].set_center((x, y))
                            collision_circles[ped_i].set_edgecolor(cmap_fake(ped_i))
                            collision_circles[ped_i].set_visible(True)

                            ## add persistent yellow collision circle
                            ax.add_artist(plt.Circle((x, y), collide_circle_rad, fc=yellow, zorder=1, ec='none'))
                            collided_delays[ped_i] = collision_delay
                            break
                        collision_circles[ped_i].set_visible(False)
                        collision_texts[ped_i].set_visible(False)

        self.update = update

        if fig is not None:
            anim = animation.FuncAnimation(fig, update, frames=obs_len + pred_len + extend_last_frame, interval=500)
            anim.save(save_fn)
            print(f"saved animation to {save_fn}")
            plt.close(fig)
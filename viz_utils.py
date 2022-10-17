import numpy as np

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches


def plot_traj_anim(obs_traj, save_fn, ped_radius=0.1, ped_discomfort_dist=0.5, pred_traj_gt=None, pred_traj_fake=None,
                   bounds=None, int_cat_abbv=None, scene_stats=None, cfg_names=None,
                   collision_mats=None, cmap_name='tab10', extend_last_frame=3, show_ped_stats=False,
                   text_time=None, text_fixed=None, grid_values=None, plot_collisions_all=True, plot_title=None):
    """
    obs_traj: shape (8, num_peds, 2) observation input to model, first 8 timesteps of the scene
    save_fn: file name where to save animation
    ped_diameter: collision threshold -- pedestrian radius * 2
    pred_traj_fake: tensor of shape (8 or 12, num_peds, 2), or list of tensors of shape (8 or 12, num_peds, 2)
                    (must be same shape as pred_traj_gt, the traj predicted by the model)
    bounds: plotting bonuds, if not specified the min and max bounds of whichever trajectories are present are used
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_title = f"{plot_title}\n" if plot_title is None else ""
    ax.set_title(f"{plot_title}{save_fn}\ninteraction_type: {int_cat_abbv}")
    ax.set_aspect("equal")

    if bounds is None:  # calculate bounds automatically
        all_traj = obs_traj
        if pred_traj_gt is not None:
            all_traj = np.concatenate([all_traj, pred_traj_gt])
        if pred_traj_fake is not None:
            if isinstance(pred_traj_fake, np.ndarray):
                all_traj = np.concatenate([all_traj, pred_traj_fake])
            elif isinstance(pred_traj_fake, list) and len(pred_traj_fake[0].shape) == 4:
                all_traj = np.concatenate([all_traj, *[p for ptf in pred_traj_fake for p in ptf]])
            elif isinstance(pred_traj_fake, list) and len(pred_traj_fake[0].shape) == 3:
                all_traj = np.concatenate([all_traj, *pred_traj_fake])
            else:
                raise NotImplementedError
        x_low, x_high = np.min(all_traj[:, :, 0]) - ped_radius, np.max(all_traj[:, :, 0]) + ped_radius
        y_low, y_high = np.min(all_traj[:, :, 1]) - ped_radius, np.max(all_traj[:, :, 1]) + ped_radius
    else:  # set bounds as specified
        x_low, x_high, y_low, y_high = bounds
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    # plotting vars
    num_peds = obs_traj.shape[1]
    obs_len = obs_traj.shape[0]
    if pred_traj_gt is not None:
        pred_len = pred_traj_gt.shape[0]
    elif pred_traj_fake is not None:
        pred_len = pred_traj_fake.shape[0]
    else:
        pred_len = 0

    # color and style properties
    delta = .32  # ped stats text offset
    text_offset_x = 0.2
    text_offset_y = 0.2
    obs_alpha = 1  # how much alpha to plot obs traj
    pred_alpha = 0.8  # how much alpha to plot gt traj, if they exist
    # each sample a different marker
    markers_0 = ['o', '*', '^', 's', '1', 'P', 'x', '$\#$', ',', '$\clubsuit$'] #'v', '<', ',', ]
    markers_1 = ['P', 'x', '$\#$', ',', '$\clubsuit$'] #'v', '<', ',', ]
    # each ped a different color
    cmap_real = plt.get_cmap(cmap_name, max(10, num_peds))
    cmap_fake = plt.get_cmap(cmap_name, max(10, num_peds))
    color_fake = [['#0D47A1', '#2196F3'],  # blue
                  ['#E65100', '#FF9800'],  # orange
                  ['#194D33', '#4CAF50'],  # green
                  ['#B71C1C', '#F44336'],  # red
                  ['#4A148C', '#9C27B0'],  # purple
                  ['#312502', '#795548'],  # brown
                  ['#b31658', '#E91E63'],  # pink
                  ['#333333', '#999999'],  # gray
                  ['#F0F4C3', '#AFB42B'],  # olive
                  ]
    # each model a different alpha and linestyle
    alpha_min = 0.3
    linestyles = ['dotted', '--']

    # add scene-related stats as descriptive text
    if show_ped_stats:
        if scene_stats is not None:
            values = map(lambda x: f"{x:0.2f}", scene_stats.values())
            scene_stats_text = f'{" / ".join(map(str, scene_stats.keys()))}\n{" / ".join(values)}'
            ax.add_artist(plt.text(x_low + 0.1, y_high + .2, scene_stats_text, fontsize=8))
            # ax.add_artist(plt.text(x_low + 0.1, y_high - .3, 'obs // pred (avg_speed / std_speed / smoothness)', fontsize=8))

    ## text that changes each frame
    text_over_time = plt.text(14, 6, "", fontsize=10, color='k', weight='bold')
    ax.add_artist(text_over_time)

    ## text that stays fixed each frame
    offset_lower = 0.1
    if isinstance(text_fixed, str):
        ax.add_artist(plt.text(x_low + offset_lower, y_low + offset_lower, text_fixed, fontsize=8))
    elif isinstance(text_fixed, list):
        text = "\n".join(text_fixed)
        ax.add_artist(plt.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=8))
    elif isinstance(text_fixed, dict):
        text = "\n".join([f'{k}: {v:0.3f}' for k, v in text_fixed.items()])
        ax.add_artist(plt.text(x_low + offset_lower, y_low + offset_lower, text, fontsize=8))
    else:
        if text_fixed is not None:
            raise NotImplementedError("text_fixed is unrecognized format")

    # ped graph elements
    circles_gt, circles_fake, last_obs_circles, lines_pred_gt, \
    lines_obs_gt, lines_pred_fake = [], [], [], [], [], []

    # plot circles to represent peds
    legend_lines = []
    legend_labels = []

    for ped_i in range(num_peds):
        color_real = cmap_real(ped_i)
        # color_fake = cmap_fake(ped_i)
        # color_real = cmap_real(0)

        ## plot ground-truth obs and pred
        circles_gt.append(ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=color_real, zorder=0)))
        line_obs_gt = mlines.Line2D(*obs_traj[0:1].T, color=color_real, marker='.', linestyle='-', linewidth=1,
                                    alpha=obs_alpha, zorder=0)
        lines_obs_gt.append(ax.add_artist(line_obs_gt))

        if pred_traj_gt is not None:
            line_pred_gt = mlines.Line2D(*obs_traj[0:1].T, color=color_real, marker='.', linestyle='-', linewidth=1,
                                         alpha=obs_alpha, zorder=0, visible=False)
            lines_pred_gt.append(ax.add_artist(line_pred_gt))

        if pred_traj_fake is not None:  # plot fake pred trajs
            if isinstance(pred_traj_fake, np.ndarray):
                pred_traj_fake = [pred_traj_fake]
            lpf, cf = [], []
            for model_i, ptf in enumerate(pred_traj_fake):
                lpf_inner, cf_inner = [], []
                # alpha = 1#0.5#(len(pred_traj_fake) - model_i - 1) * (1 - alpha_min) / len(pred_traj_fake) + alpha_min
                for sample_i, p in enumerate(ptf):
                    circle_fake = plt.Circle(ptf[0, ped_i], ped_radius, fill=True, color=color_fake[ped_i][model_i],
                                             alpha=pred_alpha, visible=False, zorder=1)
                    cf_inner.append(ax.add_artist(circle_fake))
                    label = f"{cfg_names[model_i]} ped {ped_i}" if sample_i == 0 else None
                    marker = locals()[f'markers_{model_i}'][sample_i]
                    color = color_fake[ped_i][model_i]
                    line_pred_fake = mlines.Line2D(*ptf[0:1].T, color=color,
                                                   marker=marker,
                                                   linestyle=linestyles[model_i],
                                                   alpha=pred_alpha, zorder=2,
                                                   visible=False)
                    if label is not None:
                        legend_labels.append(label)
                        legend_lines.append(patches.Patch(color=color, linestyle=linestyles[model_i], label=label))

                    lpf_inner.append(ax.add_artist(line_pred_fake))
                cf.append(cf_inner)
                lpf.append(lpf_inner)
            lines_pred_fake.append(lpf)
            circles_fake.append(cf)

    ax.legend(handles=legend_lines, loc='upper right')
    # ax.legend(legend_lines, legend_labels, loc='upper right')
    # ax.legend(legend_lines, legend_labels, loc='upper right')
    # ax.legend(handles=[line1, line2])

    # add interaction category annotations, if specified
    ped_texts = []
    for ped_i in range(num_peds):
        int_text = ax.text(circles_gt[ped_i].center[0] + text_offset_x, circles_gt[ped_i].center[1] - text_offset_y,
                           str(ped_i), color='black', fontsize=8)
        ped_texts.append(ax.add_artist(int_text))

    # plot collision circles for predictions only
    if collision_mats is not None:
        collide_circle_rad = (ped_radius)# + ped_discomfort_dist) * 2
        # assert collision_mats.shape == (pred_len, num_peds, num_peds)
        collision_circles = [ax.add_artist(plt.Circle((0, 0), collide_circle_rad, fill=False, zorder=5, visible=False))
                             for _ in range(num_peds)]
        collision_texts = [ax.add_artist(ax.text(0, 0, "", visible=False, fontsize=8)) for _ in range(num_peds)]
        collision_delay = 3
        yellow = (.8, .8, 0, .2)
        collided_delays = np.zeros(num_peds)

    # heatmap
    if grid_values is not None:
        x, y = np.meshgrid(np.linspace(*bounds[:2], grid_values.shape[1] + 1),
                           np.linspace(*bounds[2:4], grid_values.shape[2] + 1))
        # z = grid_values[0].reshape(x.shape[0] - 1, x.shape[1] - 1)
        z = grid_values[0]

        z_min, z_max = np.min(np.array(z)), np.max(np.array(z))
        state_mesh = ax.pcolormesh(x, y, z, alpha=.8, vmin=0, vmax=1, zorder=3)

    ## animation update function
    def update(frame_i):
        nonlocal x, y
        # for replicating last scene
        if frame_i >= obs_len + pred_len:
            return

        # energy text
        if frame_i < obs_len + pred_len - 1 and text_time is not None:
            energy_time, energy_time_each_ped, dmp, spd, drc, col, atr, grp = text_time[frame_i]
            energy_text = '\n'.join(f"{text_time[frame_i]:0.3f}")
            text_over_time.set_text(f"{energy_text}")

        # heatmap
        if grid_values is not None and frame_i < obs_len + pred_len - 1:
            nonlocal state_mesh, x, y
            z = grid_values[frame_i]
            normed_z = ((z - z_min) / (z_max - z_min)).reshape(x.shape[0] - 1, x.shape[1] - 1)
            state_mesh.remove()
            state_mesh = ax.pcolormesh(x, y, normed_z, alpha=.1, vmin=0, vmax=1, zorder=1)

        # move the real and pred (fake) agent
        if frame_i < obs_len:
            for ped_i, (circle_gt, line_obs_gt, ped_text) in enumerate(zip(
                    circles_gt, lines_obs_gt, ped_texts)):
                circle_gt.center = obs_traj[frame_i, ped_i]
                line_obs_gt.set_data(*obs_traj[0:frame_i + 1, ped_i].T)
                # move the pedestrian texts (ped number and relation)
                if len(ped_texts) > 0:
                    ped_text.set_position((circle_gt.center[0] + text_offset_x, circle_gt.center[1] - text_offset_y))

        elif frame_i == obs_len:
            [circle_fake.set_visible(True) for cf in circles_fake for cf_inner in cf for circle_fake in cf_inner]
            if pred_traj_gt is None:
                for circle_gt in circles_gt:
                    circle_gt.set_radius(ped_radius * 0.5)
                    circle_gt.set_alpha(0.3)
            for line_obs_gt in lines_obs_gt:
                line_obs_gt.set_alpha(0.2)
            if pred_traj_gt is not None:
                for line_pred_gt in lines_pred_gt:
                    line_pred_gt.set_visible(True)
            if pred_traj_fake is not None:
                for lpf in lines_pred_fake:
                    for lpf_inner in lpf:
                        for line_pred_fake in lpf_inner:
                            line_pred_fake.set_visible(True)

            for last_obs_circ in last_obs_circles:
                last_obs_circ.set_radius(ped_radius * 0.75)
                last_obs_circ.set_alpha(0.3)
            # obs_pred_text.set_text('prediction')

        if obs_len <= frame_i < obs_len + pred_len:
            if pred_traj_gt is not None:
                # traj_gt = np.concatenate([obs_traj, pred_traj_gt])
                for ped_i, (circle_gt, line_pred_gt, ped_text) in enumerate(zip(
                        circles_gt, lines_pred_gt, ped_texts)):
                    circle_gt.center = pred_traj_gt[frame_i - obs_len, ped_i]
                    last_obs_pred_gt = np.concatenate(
                            [obs_traj[-1:, ped_i], pred_traj_gt[0:frame_i + 1 - obs_len, ped_i]])
                    line_pred_gt.set_data(*last_obs_pred_gt.T)
                    # move the pedestrian texts (ped number and relation)
                    if len(ped_texts) > 0:
                        ped_text.set_position(
                                (circle_gt.center[0] + text_offset_x, circle_gt.center[1] - text_offset_y))

            if pred_traj_fake is not None:
                for ped_i, (cf, lpf) in enumerate(zip(circles_fake, lines_pred_fake)):
                    for model_i, (cf_inner, lpf_inner) in enumerate(zip(cf, lpf)):
                        for sample_i, (circle_fake, line_pred_fake) in enumerate(zip(cf_inner, lpf_inner)):
                            circle_fake.set_visible(True)
                            circle_fake.center = pred_traj_fake[model_i][sample_i, frame_i - obs_len, ped_i]
                            last_obs_pred_fake = np.concatenate([obs_traj[-1:, ped_i], pred_traj_fake[model_i][sample_i, 0:frame_i + 1 - obs_len, ped_i]])
                            line_pred_fake.set_data(*last_obs_pred_fake.T)

        # update collision circles (only if we are during pred timesteps)
        if (plot_collisions_all or obs_len <= frame_i <= obs_len + pred_len) and collision_mats is not None:
            if pred_traj_fake is not None:
                obs_gt_fake = np.concatenate([obs_traj, pred_traj_fake])
            else:
                obs_gt_fake = np.concatenate([obs_traj, pred_traj_gt])
            for ped_i in range(num_peds):
                # new frame; decrease the text disappearance delay by 1
                if collided_delays[ped_i] > 0:
                    collided_delays[ped_i] -= 1
                for ped_j in range(ped_i):
                    if collided_delays[ped_i] > 0:  # still in delay, circle doesn't disappear
                        break
                    elif collision_mats[frame_i-obs_len, ped_i, ped_j]:
                        ## put the center of the circle in the point between the two ped centers
                        x = (obs_gt_fake[frame_i-obs_len][ped_i][0] + obs_gt_fake[frame_i-obs_len][ped_j][0]) / 2
                        y = (obs_gt_fake[frame_i-obs_len][ped_i][1] + obs_gt_fake[frame_i-obs_len][ped_j][1]) / 2
                        collision_circles[ped_i].set_center((x, y))
                        collision_circles[ped_i].set_edgecolor(cmap_fake(ped_i))
                        collision_circles[ped_i].set_visible(True)

                        ## add persistent yellow collision circle
                        ax.add_artist(plt.Circle((x, y), collide_circle_rad, fc=yellow, zorder=1, ec='none'))
                        collided_delays[ped_i] = collision_delay
                        break
                    collision_circles[ped_i].set_visible(False)
                    collision_texts[ped_i].set_visible(False)

    anim = animation.FuncAnimation(fig, update, frames=obs_len + pred_len + extend_last_frame, interval=500)
    anim.save(save_fn)
    print(f"saved animation to {save_fn}")
    plt.close(fig)



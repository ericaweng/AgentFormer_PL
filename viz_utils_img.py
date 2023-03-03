import numpy as np

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_scene(obs_traj, save_fn=None, plot_title=None, ped_radius=0.1, gt_traj=None,
               pred_traj=None, bounds=None, collision_mats=None, text_fixed=None,
               bkg_img_path=None,
               plot_velocity_arrows=False, ax=None, agent_outline_colors=None, agent_texts=None):
    obs_ts, num_peds, _ = obs_traj.shape
    if pred_traj is not None:
        pred_len, _, _ = pred_traj.shape

    if agent_outline_colors is not None:
        assert len(agent_outline_colors) == num_peds
    if agent_texts is not None:
        assert len(agent_texts) == num_peds

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    else:
        fig = None

    plot_title = f"{plot_title}\n" if plot_title is not None else ""
    ax.set_title(plot_title, fontsize=16)
    ax.set_aspect("equal")

    if bounds is None:  # calculate bounds automatically
        all_traj = obs_traj.reshape(-1, 2)
        if gt_traj is not None:
            all_traj = np.concatenate([all_traj, gt_traj.reshape(-1, 2)])
        if pred_traj is not None:
            all_traj = np.concatenate([all_traj, pred_traj.reshape(-1, 2)])
        x_low, x_high = np.min(all_traj[:, 0]) - ped_radius, np.max(all_traj[:, 0]) + ped_radius
        y_low, y_high = np.min(all_traj[:, 1]) - ped_radius, np.max(all_traj[:, 1]) + ped_radius
    else:  # set bounds as specified
        x_low, y_low, x_high, y_high = bounds
    ax.set_xlim(x_low, x_high)
    ax.set_ylim(y_low, y_high)

    # color and style properties
    text_offset_x = -0.2
    text_offset_y = -0.2
    lw = 2
    obs_alpha = 0.2
    gt_alpha = 0.4
    cmap_name = 'tab10'
    cmap_gt = plt.get_cmap(cmap_name, max(10, num_peds))
    cmap_pred = plt.get_cmap(cmap_name, max(10, num_peds))

    circles_gt, circles_pred, last_obs_circles, lines_pred_gt, lines_obs_gt, lines_pred_fake = [], [], [], [], [], []

    # plot obs traj
    for ped_i in range(num_peds):
        obs_color = cmap_gt(ped_i % num_peds)
        # circles_gt.append( ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=obs_color,
        #                                             alpha=obs_alpha, zorder=0)))
        ax.add_artist(mlines.Line2D(*obs_traj[:, ped_i].T, color=obs_color, alpha=obs_alpha,
                                    marker=None, linestyle='-', linewidth=lw, zorder=1))
    # plot gt futures
    if gt_traj is not None:
        # gt_color = 'r'
        for ped_i in range(num_peds):
            gt_color = cmap_gt(ped_i % num_peds)
            circles_gt.append( ax.add_artist(plt.Circle(gt_traj[-1, ped_i], ped_radius, fill=True, color=gt_color,
                                                        alpha=gt_alpha, zorder=0)))
            gt_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], gt_traj[:, ped_i]])
            ax.add_artist(mlines.Line2D(*gt_plus_last_obs.T, color=gt_color, alpha=gt_alpha,
                                        marker=None, linestyle='-', linewidth=lw, zorder=1))
        # plot ped texts
        ped_texts = []
        for ped_i in range(num_peds):
            # weight = 'bold' if ped_i in highlight_peds else None
            weight = None
            int_text = ax.text(gt_traj[-1, ped_i, 0] + text_offset_x,
                               gt_traj[-1, ped_i, 1] - text_offset_y,
                               f'{ped_i}', color='black', fontsize=8, weight=weight)
            ped_texts.append(ax.add_artist(int_text))

    # plot pred futures
    if pred_traj is not None:
        # pred_fake_color = 'g'
        for ped_i in range(num_peds):
            pred_color = cmap_pred(ped_i % num_peds)
            circles_pred.append(ax.add_artist(plt.Circle(pred_traj[-1, ped_i], ped_radius, fill=True,
                                                         color=pred_color, zorder=0)))
            pred_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], pred_traj[:, ped_i]])
            ax.add_artist(mlines.Line2D(*pred_plus_last_obs.T, color=pred_color,
                                        marker=None, linestyle='--', linewidth=lw, zorder=1))
        # plot collision circles
        if collision_mats is not None:
            collide_circle_rad = (ped_radius + 0.3)
            yellow = (.8, .8, 0, .2)
            last_ts_col = False
            for t in range(pred_len):
                for ped_i in range(num_peds):
                    for ped_j in range(ped_i):
                        if collision_mats[t, ped_i, ped_j] and not last_ts_col:
                            x = (pred_traj[t][ped_i][0] + pred_traj[t][ped_j][0]) / 2
                            y = (pred_traj[t][ped_i][1] + pred_traj[t][ped_j][1]) / 2
                            ax.add_artist(plt.Circle((x, y), collide_circle_rad, fc=yellow, zorder=1, ec='none'))
                            last_ts_col = True
                        else:
                            last_ts_col = False

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


    # OTHER RANDO STUFF
    if agent_texts is not None:
        ax.add_artist(ax.text(obs_traj[-1, ped_i, 0] + text_offset_x,
                              obs_traj[-1, ped_i, 1] - text_offset_y,
                              agent_texts[ped_i], weight='bold', color='k', fontsize=8))

    small_arrows = False#True
    if plot_velocity_arrows:
        if small_arrows:
            arrow_style = patches.ArrowStyle("->", head_length=2, head_width=2)
        else:
            arrow_style = patches.ArrowStyle("->", head_length=3, head_width=3)
        # arrow_color = (1, 1, 1, 1)
        arrow_color = (0, 0, 0, 1)
        if pred_traj is not None:
            traj = np.concatenate([obs_traj, pred_traj])
        else:
            traj = obs_traj
        traj = pred_traj
        traj = obs_traj
        arrow_starts = traj[:-1].reshape(-1, 2)
        arrow_ends = arrow_starts + 0.1 * (traj[1:] - traj[:-1]).reshape(-1, 2)
        for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
            ax.add_artist(patches.FancyArrowPatch(arrow_start, arrow_end, color=arrow_color,
                                                  arrowstyle=arrow_style, zorder=obs_ts + 1,
                                                  linewidth=0.4 if small_arrows else 1.5))
        last_arrow_starts = traj[-1]
        last_arrow_ends = traj[-1] + 0.1 * (traj[-1] - traj[-2]).reshape(-1, 2)
        for arrow_start, arrow_end in zip(last_arrow_starts, last_arrow_ends):
            ax.add_artist(patches.FancyArrowPatch(arrow_start, arrow_end,
                                                  color=arrow_color,  # 'r',
                                                  arrowstyle=arrow_style,
                                                  zorder=obs_ts + 10,
                                                  linewidth=0.4 if small_arrows else 2))

    if fig is not None:
        if save_fn is not None:
            plt.savefig(save_fn)
            print(f"saved image to {save_fn}")
        plt.close(fig)


def plot_img_grid(save_fn, title=None, bounds=None, plot_size=None, *list_of_arg_dicts):
    """plot_fn takes in a single arg_dict and an argument, ax, and plots on the ax"""
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
    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(7.5 * num_plots_width, 5 * num_plots_height))
    if isinstance(axes[0], np.ndarray):
        axes = [a for ax in axes for a in ax]

    for ax_i, (arg_dict, ax) in enumerate(zip(list_of_arg_dicts, axes)):
        plot_fn = plot_scene
        plot_fn(**arg_dict, ax=ax, bounds=bounds)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)
    if title is not None:
        fig.suptitle(title, fontsize=16)
    fig.savefig(save_fn)
    print(f"saved figure to {save_fn}")
    plt.close(fig)

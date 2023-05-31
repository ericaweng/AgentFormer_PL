import numpy as np

from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_scene(obs_traj, save_fn=None, plot_title=None, ped_radius=0.1, gt_traj=None, sample_i=None,
               is_best_ADE_of_all=None, is_best_JADE_of_all=None, is_best_JADE=None, x_label=None, y_label=None, ade_is=None,
               pred_traj=None, bounds=None, collision_mats=None, text_fixed_bl=None, text_fixed_br=None, subtitle=None,
               bkg_img_path=None, text_fixed_tr=None, text_fixed_tl=None, highlight_peds=None, plot_ped_texts=True,
               plot_velocity_arrows=False, ax=None, agent_outline_colors=None, agent_texts=None,
               fig=None):
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

    fp = mpl.font_manager.FontProperties(fname="/root/code/HelveticaNeueBold.ttf")
    ax.set_title(plot_title, fontproperties=fp, fontsize=30)
    JADE_FS = 22
    SUBTIT_FS = 18
    ORANGE = '#fb8072'#'#D16D00'
    YELLOW = '#9E9849'
    GRAY = '#cccccc'
    BLACK = 'k'
    if subtitle is not None:
        subtitle_l, subtitle_r = subtitle.split('   ')
        if is_best_JADE_of_all:
            color_l = ORANGE
            subtitle_l += ' (Best)'
        else:
            color_l = GRAY
        if is_best_ADE_of_all:
            color_r = ORANGE#YELLOW
            subtitle_r += ' (Best)'
        else:
            color_r = GRAY
        subtitle_offset = 0.5
        subtitle_offset_y = 0.2
        additional_r = 1.0 if is_best_ADE_of_all else 0
        additional_l = 1.0 if is_best_JADE_of_all else 0
        ax.text((x_low + x_high)/2 - subtitle_offset + additional_l - additional_r,
                y_high + subtitle_offset_y, subtitle_l, fontproperties=fp, fontsize=SUBTIT_FS, ha='right', color=color_l)
        ax.text((x_low + x_high)/2 + subtitle_offset - additional_r + additional_l,
                y_high + subtitle_offset_y, subtitle_r, fontproperties=fp, fontsize=SUBTIT_FS, ha='left', color=color_r)
    # ax.text((x_low + x_high)/2, y_high + 0.5, subtitle, fontproperties=fp, fontsize=16, ha='center')
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=16, fontproperties=fp)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=24, fontproperties=fp)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # color and style properties
    text_offset_x = -0.5
    text_offset_y = -0.2
    lw = 4
    obs_alpha = 1#0.2
    gt_alpha = 0.4
    pred_alpha = 0.5
    cmap_name = 'tab10'
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
    cmap_gt = cmap_pred = lambda i: colors[i]

    circles_gt, circles_pred, last_obs_circles, lines_pred_gt, lines_obs_gt, lines_pred_fake = [], [], [], [], [], []

    # plot obs traj
    for ped_i in range(num_peds):
        obs_color = cmap_gt(ped_i % len(colors))
        circles_gt.append( ax.add_artist(plt.Circle(obs_traj[0, ped_i], ped_radius, fill=True, color=obs_color,
                                                    alpha=obs_alpha, zorder=0)))
        ax.add_artist(mlines.Line2D(*obs_traj[:, ped_i].T, color=obs_color, alpha=obs_alpha,
                                    marker=None, linestyle='-', linewidth=lw, zorder=1))
    # plot gt futures
    if gt_traj is not None:
        # gt_color = 'r'
        for ped_i in range(num_peds):
            gt_color = cmap_gt(ped_i % len(colors))
            # circles_gt.append( ax.add_artist(plt.Circle(gt_traj[-1, ped_i], ped_radius, fill=True, color=gt_color,
            #                                             alpha=gt_alpha, zorder=0)))
            gt_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], gt_traj[:, ped_i]])
            ax.add_artist(mlines.Line2D(*gt_plus_last_obs.T, color=gt_color, alpha=gt_alpha,
                                        marker=None, linestyle='-', linewidth=lw, zorder=1))
        # plot ped texts
        if plot_ped_texts:
            ped_texts = []
            for ped_i in range(num_peds):
                weight = 'bold' if highlight_peds is not None and ped_i in highlight_peds else None
                # weight = None
                int_text = ax.text(gt_traj[-1, ped_i, 0] + text_offset_x,
                                   gt_traj[-1, ped_i, 1] - text_offset_y,
                                   f'A{ped_i}', color='black', fontsize=14, weight=weight)
                # int_text = ax.text(pred_traj[-1, ped_i, 0] + text_offset_x,
                #                    pred_traj[-1, ped_i, 1] - text_offset_y,
                #                    f'A{ped_i}', color='black', fontsize=14, weight=weight)
                ped_texts.append(ax.add_artist(int_text))

    # plot pred futures
    if pred_traj is not None:
        for ped_i in range(num_peds):
            pred_color = cmap_pred(ped_i % len(colors))
            pred_plus_last_obs = np.concatenate([obs_traj[-1:,ped_i], pred_traj[:, ped_i]])
            ax.add_artist(mlines.Line2D(*pred_plus_last_obs.T, color=pred_color, alpha=pred_alpha,
                                        marker=None, linestyle='--', linewidth=lw, zorder=1))
        # plot collision circles
        if collision_mats is not None:
            collide_circle_rad = (ped_radius + 0.8)
            YELLOW_COL = (.9, .3, 0, .2)
            # YELLOW_COL = (.9, .9, 0, .3)
            last_ts_cols = np.zeros((num_peds, num_peds))
            for t in range(pred_len):
                for ped_i in range(num_peds):
                    for ped_j in range(ped_i):
                        if collision_mats[t, ped_i, ped_j] and last_ts_cols[ped_i, ped_j] <= 0:
                            x = (pred_traj[t][ped_i][0] + pred_traj[t][ped_j][0]) / 2
                            y = (pred_traj[t][ped_i][1] + pred_traj[t][ped_j][1]) / 2
                            ax.add_artist(plt.Circle((x, y), collide_circle_rad, fc=YELLOW_COL, zorder=1, ec='none'))
                            last_ts_cols[ped_i, ped_j] = 3
                        last_ts_cols[ped_i, ped_j] -= 1

    texts_fixed = [text_fixed_tr, text_fixed_tl, text_fixed_br, text_fixed_bl]
    offset = 0.3
    offsets = [(x_high - offset, y_high - offset), (x_low + offset, y_high - offset),
               (x_low + offset, y_high + offset), (x_high - offset, y_high + offset)]
    va_has = [('top', 'right'), ('top', 'left'), ('bottom', 'right'), ('bottom', 'left')]
    for text_i, (text_fixed, (pos_x, pos_y), (va, ha)) in enumerate(zip(texts_fixed, offsets, va_has)):
        if text_fixed is None:
            continue
        if text_i == 0:
            text_fixed_fs = JADE_FS
            if is_best_JADE:
                color = ORANGE
                text_fixed += " (Best)"
            else:
                color = GRAY
            text = ax.text(pos_x, pos_y, text_fixed, fontsize=text_fixed_fs, color=color, fontproperties=fp, va=va,
                           ha=ha)
            ax.add_artist(text)
        elif text_i == 1:
            text_fixed_fs = 16
            color = GRAY
            texts = text_fixed.split('\n')[1:]
            # rc('text', usetex=True)
            text = ax.text(pos_x, pos_y, 'ADE:', weight='bold', fontsize=text_fixed_fs, color=color, fontproperties=fp, va=va, ha=ha)
            ax.add_artist(text)
            # rc('text', usetex=False)
            TEXT_HEIGHT = 0.4
            for text_i, ade_text in enumerate(texts):
                if sample_i == ade_is[text_i]:
                    color = ORANGE#BLACK#YELLOW
                    ade_text += " (Best)"
                else:
                    color = GRAY
                text = ax.text(pos_x, pos_y - (text_i + 1) * TEXT_HEIGHT, ade_text, fontsize=text_fixed_fs, color=color, fontproperties=fp, va=va, ha=ha)
                ax.add_artist(text)
        else:
            raise NotImplementedError("tbd")


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
            # arrow_style = patches.ArrowStyle("-|>", head_length=5, head_width=3)
            arrow_style = patches.ArrowStyle("-|>", head_length=10, head_width=5)
        # arrow_color = (1, 1, 1, 1)
        arrow_color = (0, 0, 0, 1)
        if pred_traj is not None:
            traj = np.concatenate([obs_traj, pred_traj])
        else:
            traj = obs_traj
        # traj = pred_traj
        traj = obs_traj  # only plot arrows on obs
        # arrow_starts = traj[:-1].reshape(-1, 2)
        # arrow_ends = arrow_starts + 0.1 * (traj[1:] - traj[:-1]).reshape(-1, 2)
        # for arrow_start, arrow_end in zip(arrow_starts, arrow_ends):
        #     ax.add_artist(patches.FancyArrowPatch(arrow_start, arrow_end, color=arrow_color,
        #                                           arrowstyle=arrow_style, zorder=obs_ts + 1,
        #                                           linewidth=0.4 if small_arrows else 1.5))
        last_arrow_starts = traj[-1]
        last_arrow_ends = traj[-1] + 0.1 * (traj[-1] - traj[-2]).reshape(-1, 2)
        last_arrow_lengths = 0.1 * (traj[-1] - traj[-2]).reshape(-1, 2)
        for ped_i, (arrow_start, arrow_end, arrow_length) in enumerate(zip(last_arrow_starts, last_arrow_ends, last_arrow_lengths)):
            arrow_color = cmap_pred(ped_i % len(colors))
            ax.add_artist(patches.FancyArrow(*arrow_start, *arrow_length,
                                             overhang=3,
                                             head_width=.5,
                                             head_length=.2,
                                             color=arrow_color,  # 'r',
                                             # arrowstyle=arrow_style,
                                             zorder=0,
                                             linewidth=0.4 if small_arrows else 2))
            # ax.add_artist(patches.FancyArrowPatch(arrow_start, arrow_end,
            #                                       color=arrow_color,  # 'r',
            #                                       arrowstyle=arrow_style,
            #                                       zorder=obs_ts + 10,
            #                                       linewidth=0.4 if small_arrows else 2))

    if fig is not None:
        if save_fn is not None:
            plt.savefig(save_fn)
            print(f"saved image to {save_fn}")
        plt.close(fig)

def plot_img_grid_new(save_fn, title=None, bounds=None, plot_size=None, transposed=False, scale_x=6.5, additional_y=0, list_of_arg_dicts=None):
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

    scale_y = (bounds[3] - bounds[1]) / (bounds[2] - bounds[0]) * scale_x
    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(scale_x * num_plots_width, scale_y * num_plots_height + additional_y))

    if len(list_of_arg_dicts) == 1:
        axes = [axes]
    if len(list_of_arg_dicts[0]) == 1:
        axes = [[a] for a in axes]
    assert len(axes) == len(list_of_arg_dicts), f'axes ({len(axes)}) and list_of_arg_dicts ({len(list_of_arg_dicts)}) must have same length'
    assert len(axes[0]) == len(list_of_arg_dicts[0]), f'axes ({len(axes[0])}) and list_of_arg_dicts ({len(list_of_arg_dicts[0])}) must have same length'
    if transposed:
        for ax_i, axs in enumerate(axes):
            for ax_j, ax in enumerate(axs):
                plot_fn = plot_scene
                arg_dict = list_of_arg_dicts[ax_j][ax_i]
                plot_fn(**arg_dict, ax=ax, bounds=bounds)
    else:
        for ax_i, axs in enumerate(axes):
            for ax_j, ax in enumerate(axs):
                plot_fn = plot_scene
                arg_dict = list_of_arg_dicts[ax_i][ax_j]
                plot_fn(**arg_dict, ax=ax, bounds=bounds)

    fig.tight_layout()
    hspace = 0 if transposed else 0.3
    fig.subplots_adjust(top=0.98, bottom=0.01, hspace=hspace, wspace=-0.00)
    # fig.subplots_adjust(hspace=0.6, wspace=-0.0)
    if title is not None:
        fig.suptitle(title, fontsize=36)
    fig.savefig(save_fn)
    print(f"saved figure to {save_fn}")
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

    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(6.5 * num_plots_width, 2.5 * num_plots_height))

    for ax_i, axs in enumerate(axes):
        for ax_j, ax in enumerate(axs):
            plot_fn = plot_scene
            arg_dict = list_of_arg_dicts[ax_j][ax_i]
            plot_fn(**arg_dict, ax=ax, bounds=bounds)

    fig.tight_layout()
    fig.subplots_adjust(top=0.96, bottom=0.01, hspace=0.0, wspace=-0.00)
    # fig.subplots_adjust(hspace=0.6, wspace=-0.0)
    if title is not None:
        fig.suptitle(title, fontsize=36)
    fig.savefig(save_fn)
    print(f"saved figure to {save_fn}")
    plt.close(fig)


def plot_img_grid_old(save_fn, title=None, bounds=None, plot_size=None, *list_of_arg_dicts):
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
    # fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(5.5 * num_plots_width, 5.5 * num_plots_height))
    # fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(5.5 * num_plots_width, 7 * num_plots_height))
    fig, axes = plt.subplots(num_plots_height, num_plots_width, figsize=(5.5 * num_plots_width, 4.0 * num_plots_height))
    if isinstance(axes[0], np.ndarray):
        axes = [a for ax in axes for a in ax]
    if isinstance(list_of_arg_dicts[0], list):
        list_of_arg_dicts = [a for arg_dict in list_of_arg_dicts for a in arg_dict]

    for ax_i, (arg_dict, ax) in enumerate(zip(list_of_arg_dicts, axes)):
        plot_fn = plot_scene
        plot_fn(**arg_dict, ax=ax, bounds=bounds)

    fig.tight_layout()
    fig.subplots_adjust(top=0.98, bottom=0.01, hspace=0.05, wspace=-0.0)
    # fig.subplots_adjust(hspace=0.6, wspace=-0.0)
    if title is not None:
        fig.suptitle(title, fontsize=36)
    fig.savefig(save_fn)
    print(f"saved figure to {save_fn}")
    plt.close(fig)

import numpy as np

from matplotlib import rc
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_scene(obs_traj, save_fn=None, plot_title=None, ped_radius=0.1, gt_traj=None, sample_i=None,
               x_label=None, y_label=None, ade_is=None,
               pred_traj=None, bounds=None, collision_mats=None, text_fixed_bl=None, text_fixed_br=None, subtitle=None,
               bkg_img_path=None, text_fixed_tr=None, text_fixed_tl=None, highlight_peds=None, plot_ped_texts=True,
               plot_velocity_arrows=False, ax=None, agent_outline_colors=None, agent_texts=None,
               ):
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

    if bkg_img_path is not None:
        try:
            import matplotlib.image as mpimg
            alpha = int(.7 * 255)
            img = mpimg.imread(bkg_img_path)
            img = np.dstack((img, alpha * np.ones_like(img[:, :, 0:1])))
            ax.imshow(img, zorder=-100)
        except FileNotFoundError:
            import ipdb; ipdb.set_trace()
            pass

    fp = mpl.font_manager.FontProperties(fname="/root/code/HelveticaNeueBold.ttf")
    ax.set_title(plot_title, fontproperties=fp, fontsize=30)
    JADE_FS = 22
    SUBTIT_FS = 18
    ORANGE = '#fb8072'#'#D16D00'
    YELLOW = '#9E9849'
    GRAY = '#cccccc'
    BLACK = 'k'
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
        for ped_i in range(num_peds):
            gt_color = cmap_gt(ped_i % len(colors))
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
                                             zorder=0,
                                             linewidth=0.4 if small_arrows else 2))

    if fig is not None:
        if save_fn is not None:
            plt.savefig(save_fn)
            print(f"saved image to {save_fn}")
        plt.close(fig)

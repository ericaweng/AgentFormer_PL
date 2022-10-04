import os
import numpy as np
import pandas as pd
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

label_to_sfm_weight = {
        "eth_agentformer_sfm_pre6": 1,
        "eth_agentformer_sfm_pre7": 3,
        "eth_agentformer_sfm_pre8-2": 10,
        "eth_agentformer_sfm_pre8-2-1": 5
}
label_to_description = {
        "eth_agentformer_sfm_pre": "sigma_d: 1.0; beta: 1.2",
        "eth_agentformer_sfm_pre2": "sigma_d: 1.0; beta: 1.2; don't use w in energy computation; loss_reduce mean (instead of sum)",
        "eth_agentformer_sfm_pre3": "don't use sfm feature in future encoder; still use it in the past encoder",
        "eth_agentformer_sfm_pre4": "input_norm_type running_norm in all modules",
        "eth_agentformer_sfm_pre4_dlow": "input_norm_type running_norm in all modules",
        "eth_agentformer_sfm_pre5": "add loss term to pre4, weight 10",
        "eth_agentformer_sfm_pre5_dlow": "add loss term to pre4, weight 10, dlow",
        "eth_agentformer_sfm_pre5_dlow1": "add loss term to pre4, weight 10, dlow",
        "eth_agentformer_sfm_pre5_dlow2": "add loss term to pre4, weight 10, dlow",
        "eth_agentformer_sfm_pre5_dlow3": "add loss term to pre4, weight 10, dlow",
        "eth_agentformer_sfm_pre5_dlow4": "add loss term to pre4, weight 10, dlow",
        "eth_agentformer_sfm_pre6": "weight 1",
        "eth_agentformer_sfm_pre7": "weight 3",
        "eth_agentformer_sfm_pre8-2": "weight 10; no feat",
        "eth_agentformer_sfm_pre8-2_dlow": "weight 10; no feat",
        "eth_agentformer_sfm_pre8-2-1": "weight 5; no feat",
        "eth_agentformer_sfm_pre8-2-1_dlow": "weight 5; no feat",
}

def main0():
    """incomplete function, just to get Ye's SFM feat ETH experiment results"""
    metrics_dir = "metrics"
    filename = f'metrics_eth12.csv'
    metrics_path = os.path.join(metrics_dir, filename)
    data = pd.read_csv(metrics_path)

    # make new cols
    sfm_data = data[data['label'].apply(lambda r: 'sfm' in r)]
    sfm_data = sfm_data[sfm_data['label'].apply(lambda label: label in label_to_description)]
    sfm_data['description'] = sfm_data['label'].apply(lambda label: label_to_description[label])
    sfm_data['experiment label'] = sfm_data['label']

    # get min ADE run of each experiment
    sfm_data = sfm_data[sfm_data.groupby(['label'])['ADE'].transform(min) == sfm_data['ADE']]

    # print data
    data_to_print = sfm_data[['experiment label','description','ADE','FDE','CR_pred','CR_pred_mean']]
    print("data_to_print:\n", data_to_print)
    data_to_print.to_csv('viz/sf_feat_ye.tsv', index=False, sep='\t')
    # col_names = ['experiment label','description','ADE','FDE','CR_pred']
    # print("\t".join(col_names))
    # for row_vals,row_name in zip(data_to_print.to_numpy(),col_names):
    #     print('\t'.join(map(str,row_vals))) #[col_names]+'\t')
    import ipdb; ipdb.set_trace()
    # sfm_data.to_csv(header=None, index=False, sep='\t')


def main():
    """plot AF+SFM hyperparam search results"""
    metrics_dir = "metrics"  # /home/yyuan2/Documents/repo/AgentFormerSDD/

    metrics_path = os.path.join(metrics_dir, "metrics_zara212.csv")
    data = pd.read_csv(metrics_path)
    data = data[data['label'].apply(lambda r: 'zara2_sfm_base' in r)]
    data['color_val_ADE'] = (data['ADE'] - data['ADE'].min()) / (data['ADE'].max() - data['ADE'].min())
    data['color_val_FDE'] = (data['FDE'] - data['FDE'].min()) / (data['FDE'].max() - data['FDE'].min())
    data['color_val_CR_pred'] = (data['CR_pred'] - data['CR_pred'].min()) / (data['CR_pred'].max() - data['CR_pred'].min())
    data['weight'] = data['label'].apply(lambda r: r.split('_')[-3].split('-')[-1])
    data['sigma_d'] = data['label'].apply(lambda r: r.split('_')[-1].split('-')[-1])
    # data['CR_p'] = data['CR_pred'] / 100
    print("data:", data)

    cmap = plt.get_cmap('coolwarm')
    import ipdb; ipdb.set_trace()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 16))
    circles = []
    rad = 1
    for row in data:
        circles.append(ax.add_artist(patches.Circle((row['weight'], row['sigma_d']), rad, fill=True, color=cmap(data['color_val_ADE']), zorder=0)))
        circles.append(ax2.add_artist(patches.Circle((row['weight'], row['sigma_d']), rad, fill=True, color=cmap(data['color_val_CR_pred']), zorder=0)))

    # for p_i, p in enumerate(chart.patches):
    #     chart.annotate(f"{df['N'][p_i]:.0f} / {df['prop'][p_i]:0.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
    #                    ha='center', va='center', fontsize=6, color='black', xytext=(0, 5),
    #                    textcoords='offset points')
    # sns.despine(bottom=True, left=True)
    # chart.set_xticklabels(chart.get_xticklabels(), visible=False)
    # chart2.set_xticklabels(chart2.get_xticklabels(), visible=False)
    # fig.legend(chart.containers[0], list(label_to_description.values()),
    #            loc='upper right', fontsize=9, labelspacing=0.25)  # , ncol=2, labelspacing=0.8)

    # fig.suptitle(f"AF+SFM's SFM vs. ADE and CR", fontsize=16)
    fig.suptitle(f"SF weight and sigma_d vs. ADE (top), CR (bottom)", fontsize=16)
    ax.set_xlabel("weight")
    ax2.set_xlabel("weight")
    ax.set_ylabel("sigma_d")
    ax2.set_ylabel("sigma_d")
    # plt.tight_layout()
    # plt.subplots_adjust(hspace=0.5)
    fig_path = f'viz/af_sfm_feat.pdf'
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close(fig)
    print(f"saved figure to {fig_path}")
    exit()


    # import ipdb; ipdb.set_trace()


def main1():
    """plot bar graph results ADE from Ye's SFM feature experiments"""
    metrics_dir = "metrics"  # /home/yyuan2/Documents/repo/AgentFormerSDD/
    env_names = ['eth', 'hotel', 'zara1', 'zara2', 'trajnet_sdd', 'univ']

    metric_filenames = [f'metrics_{env_name}12.csv' for env_name in env_names]
    for filename in metric_filenames:
        metrics_path = os.path.join(metrics_dir, filename)
        # data = np.genfromtxt(metrics_path, delimiter=',', )
        data = pd.read_csv(metrics_path)

        # get only sfm runs
        sfm_data = data[data['label'].apply(lambda r: 'sfm' in r)]

        # get only 10th epoch (because so far we don't have better)
        sfm_data = sfm_data[sfm_data['epoch'] == 10]
        # confirm CR == ACFL
        # sfm_data['CR_from_ACFL'] = (100 - sfm_data['ACFL']) / 100
        # decimal value for collision rate
        sfm_data['CR_p'] = sfm_data['CR_pred'] / 100
        # filter out rows which don't have sfm weight info
        sfm_data = sfm_data[sfm_data['label'].apply(lambda label: label in label_to_description)]
        # get weight info from either label, or label-to-weight map
        sfm_data['weight'] = sfm_data['label'].apply(
            lambda label: 0 if label[-1] == 'e' else int(label[-1]))

        # sfm_data = data[data['label'].apply(lambda r:'sfm' in r)]
        # best_model = sfm_data[sfm_data['ADE']==sfm_data['ADE'].min()]
        # print("best_model:\n", best_model)
        sfm_data_sorted = sfm_data.sort_values(by='weight')
        # sfm_data_sorted = sfm_data.sort_values(by='ADE')
        print("sfm_data_sorted:\n", sfm_data_sorted)

        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 16))
        # ax2 = ax.twinx()
        chart = sns.barplot(x='label', y='ADE', data=sfm_data_sorted, ax=ax, palette='tab10')
        chart2 = sns.barplot(x='label', y='CR_p', data=sfm_data_sorted, ax=ax2, palette='tab10')
        # for p_i, p in enumerate(chart.patches):
        #     chart.annotate(f"{df['N'][p_i]:.0f} / {df['prop'][p_i]:0.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
        #                    ha='center', va='center', fontsize=6, color='black', xytext=(0, 5),
        #                    textcoords='offset points')
        # sns.despine(bottom=True, left=True)
        chart.set_xticklabels(chart.get_xticklabels(), visible=False)
        chart2.set_xticklabels(chart2.get_xticklabels(), visible=False)
        fig.legend(chart.containers[0], list(label_to_description.values()),
                   loc='upper right', fontsize=9, labelspacing=0.25)  # , ncol=2, labelspacing=0.8)

        # fig.suptitle(f"AF+SFM's SFM vs. ADE and CR", fontsize=16)
        fig.suptitle(f"AF+SFM's different params  vs. ADE and CR", fontsize=16)
        ax.set_xlabel("parameter set")
        ax2.set_xlabel("parameter set")
        ax.set_ylabel("ADE")
        ax2.set_ylabel("collision rate")
        # plt.tight_layout()
        # plt.subplots_adjust(hspace=0.5)
        fig_path = f'viz/af_sfm_feat.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"saved figure to {fig_path}")
        exit()
        # import ipdb; ipdb.set_trace()


def main2():
    """plot line graph results from Ye's ETH SFM experiments plus my new experiments: ADE vs. weight"""
    metrics_dir = "metrics" #/home/yyuan2/Documents/repo/AgentFormerSDD/
    env_names = ['eth', 'hotel', 'zara1', 'zara2', 'trajnet_sdd', 'univ']

    metric_filenames = [f'metrics_{env_name}12.csv' for env_name in env_names]
    for filename in metric_filenames:
        metrics_path = os.path.join(metrics_dir, filename)
        # data = np.genfromtxt(metrics_path, delimiter=',', )
        data = pd.read_csv(metrics_path)

        # get only sfm runs
        sfm_data = data[data['label'].apply(lambda r:'sfm' in r)]

        # get only 10th epoch (because so far we don't have better)
        sfm_data = sfm_data[sfm_data['epoch'] == 10]
        # confirm CR == ACFL
        # sfm_data['CR_from_ACFL'] = (100 - sfm_data['ACFL']) / 100
        # decimal value for collision rate
        sfm_data['CR_p'] = sfm_data['CR_pred'] / 100
        # filter out rows which don't have sfm weight info
        # sfm_data = sfm_data[sfm_data['label'].apply(lambda label: 'agentformer' not in label)]
        sfm_data = sfm_data[sfm_data['label'].apply(lambda label: label in label_to_sfm_weight or 'agentformer' not in label)]
        # get weight info from either label, or label-to-weight map
        sfm_data['weight'] = sfm_data['label'].apply(lambda label: label_to_sfm_weight.get(label) if 'agentformer' in label else float(label.split("_")[-1]))
        # sfm_data['weight'] = sfm_data['label'].apply(lambda label: label_to_sfm_weight.get(label) if 'agentformer' in label else float(label[-2:]) if label[-2] == '.' or label[-2] else float(label[-1]))

        # sfm_data = data[data['label'].apply(lambda r:'sfm' in r)]
        # best_model = sfm_data[sfm_data['ADE']==sfm_data['ADE'].min()]
        # print("best_model:\n", best_model)
        sfm_data_sorted = sfm_data.sort_values(by='weight')
        # sfm_data_sorted = sfm_data.sort_values(by='ADE')
        print("sfm_data_sorted:\n", sfm_data_sorted)

        fig, ax = plt.subplots(1,1, figsize=(10,8))
        ax2 = ax.twinx()
        chart = sns.lineplot(x='weight', y='ADE', data=sfm_data_sorted, ax=ax, color='r')#palette='tab20')
        for x, y in zip(sfm_data_sorted['weight'], sfm_data_sorted['ADE']):
            # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
            ax.text(x=x,  # x-coordinate position of data label
                     y=y,  # y-coordinate position of data label, adjusted to be 150 below the data point
                     s='{:.2f}'.format(y),  # data label, formatted to ignore decimals
                     color='r')  # set colour of line
        chart = sns.lineplot(x='weight', y='CR_p', data=sfm_data_sorted, ax=ax2, palette='tab10')
        for x, y in zip(sfm_data_sorted['weight'], sfm_data_sorted['CR_p']):
            # the position of the data label relative to the data point can be adjusted by adding/subtracting a value from the x &/ y coordinates
            ax2.text(x=x,  # x-coordinate position of data label
                     y=y,  # y-coordinate position of data label, adjusted to be 150 below the data point
                     s='{:.2f}'.format(y),  # data label, formatted to ignore decimals
                     color='b')  # set colour of line
        # print(sfm_data_sorted['CR_p'])
        # chart = sns.barplot(x='label', y='ADE', data=sfm_data_sorted, ax=ax, palette='tab20')
        # for p_i, p in enumerate(chart.patches):
        #     chart.annotate(f"{df['N'][p_i]:.0f} / {df['prop'][p_i]:0.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
        #                    ha='center', va='center', fontsize=6, color='black', xytext=(0, 5),
        #                    textcoords='offset points')
        # sns.despine(bottom=True, left=True)
        # chart.set_xticklabels(chart.get_xticklabels(), visible=False)
        # fig.legend(chart.containers[0], list(INTERACTION_CAT_NAMES.values()),
        #            loc='lower right', fontsize=9, labelspacing=0.25)  # , ncol=2, labelspacing=0.8)

        fig.suptitle(f"AF+SFM's SFM weight vs. ADE", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(hspace=0.5)
        fig_path = f'viz/af_sfm.pdf'
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)
        print(f"saved figure to {fig_path}")
        exit()
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
    # main2()
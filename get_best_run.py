import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
        "eth_agentformer_sfm_pre5": "add loss term to pre4, weight 10",
        "eth_agentformer_sfm_pre6": "weight 1",
        "eth_agentformer_sfm_pre7": "weight 3",
        "eth_agentformer_sfm_pre8-2": "weight 10; no feat",
        "eth_agentformer_sfm_pre8-2-1": "weight 5; no feat",
}

def main():
    metrics_dir = "metrics"  # /home/yyuan2/Documents/repo/AgentFormerSDD/
    env_names = ['eth', 'hotel', 'zara1', 'zara2', 'trajnet_sdd', 'univ']

    metric_filenames = [f'metrics_{env_name}12.csv' for env_name in env_names]
    for filename in metric_filenames:
        metrics_path = os.path.join(metrics_dir, filename)
        # data = np.genfromtxt(metrics_path, delimiter=',', )
        data = pd.read_csv(metrics_path)
        data = data[data['label'] == 'eth_agentformer_pre']
        data['CR_p'] = data['CR_pred'] / 100
        best_model = data[data['ADE']==data['ADE'].min()]
        print("data:", data)

        import ipdb; ipdb.set_trace()
        # get only sfm runs
        # sfm_data = data[data['label'].apply(lambda r: 'sfm' in r)]


def main1():
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
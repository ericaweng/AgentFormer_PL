"""get best sfm sweep run"""

import glob
import argparse
import torch
import pandas as pd


def main(args):
    table = None
    nk_results = glob.glob('../trajectory_reward/results/trajectories/test_results/zara2_agentformer_nocol.tsv') + \
                 glob.glob('../trajectory_reward/results/trajectories/test_results/*nk*.tsv')

    for tsv_file in nk_results:
        # check if results have already been computed
        cfg = tsv_file.split('.tsv')[0].split('/')[-1]
        if table is None:
            table = pd.read_csv(tsv_file, sep='\t', names=['metric', 'value'])
            table['method'] = cfg
        else:
            new_table = pd.read_csv(tsv_file, sep='\t', names=['metric', 'value'])
            new_table['method'] = cfg
            table = pd.concat([table, new_table])
    table = table.reset_index()
    table = table.drop(columns=['index'])
    table = table.pivot_table(index='method', columns='metric', values='value')
    table = table.drop(columns=['total_peds', 'epoch'])
    print(table)
    s = table.style.format("{:.3f}", na_rep='-')
    latex_save_path = f'../trajectory_reward/results/trajectories/test_results/dlow_nk.tex'
    s.to_latex(latex_save_path, column_format=f"c|{'c'*len(table.columns)}", position="h",
               position_float="centering", hrules=True, label="table:5", caption="number of mods methods",
               multirow_align="t", multicol_align="r")


def sfm_sweep(args):
    for jr_or_not in ['_jr', '']:
        table = None
        for tsv_file in glob.glob('../trajectory_reward/results/trajectories/test_results/*sfm_*s-*.tsv'):
            # check for all non-joint-results right now
            if jr_or_not == '' and 'jr' in tsv_file:
                continue
            if jr_or_not == '_jr' and 'jr' not in tsv_file:
                continue
            # check if results have already been computed
            cfg = tsv_file.split('.tsv')[0].split('/')[-1]
            if len(glob.glob(f'../trajectory_reward/results/trajectories/test_results/{cfg}.tsv')) == 0:
                print(f"cfg {cfg} results has NOT already been computed")
                continue
            weight = float(cfg.split('_')[-2].split('-')[-1])
            sigma_d = float(cfg.split('_')[-1].split('-')[-1])
            if table is None:
                table = pd.read_csv(tsv_file, sep='\t', names=['metric', 'value'])
                table['method'] = cfg
                table['weight'] = f"{weight:3.1f}"
                table['sigma_d'] = f"{sigma_d:.2f}"
            else:
                new_table = pd.read_csv(tsv_file, sep='\t', names=['metric', 'value'])
                new_table['method'] = cfg
                new_table['weight'] = f"{weight:3.1f}"
                new_table['sigma_d'] = f"{sigma_d:.2f}"
                table = pd.concat([table, new_table])
        table = table.reset_index()
        num_sigma_ds = len(table['sigma_d'].unique())
        print("num_sigma_ds:", num_sigma_ds)
        table = table.drop(columns=['index'])
        table = table.groupby(['metric'])
        latex = ""

        for metric, metric_grp in table:
            if metric == 'total_peds' or metric == 'epoch':
                continue
            df = metric_grp.drop(columns=['method', 'metric'])
            df = df.pivot_table(index='weight', columns='sigma_d', values='value')
            s = df.style.format("{:.3f}", na_rep='-')
            caption_latex = metric.replace('_', '\_')
            latex += f"% ====================== {metric} ======================\n"
            latex += s.to_latex(column_format=f"c|{'c'*num_sigma_ds}", position="h", position_float="centering",
                                hrules=True, label="table:5", caption=f"{caption_latex}",
                                multirow_align="t", multicol_align="r")
            latex += '\n\n'

        latex_save_path = f'../trajectory_reward/results/trajectories/test_results/sfm{jr_or_not}_sweep.tex'
        with open(latex_save_path, 'w') as f:
            f.write(latex)
        print(table)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_cmds', '-mc', type=int, default=100)
    argparser.add_argument('--start_from', '-sf', type=int, default=0)
    argparser.add_argument('--gpus_available', '-ga', nargs='+', type=int,
                           default=list(range(torch.cuda.device_count())))
    argparser.add_argument('--trial', '-t', action='store_true')
    main(argparser.parse_args())
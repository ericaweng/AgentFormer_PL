"""test models nocol
and train cmd"""
import multiprocessing
from itertools import chain
import os
import glob
from pathlib import Path
import torch
import argparse

from traj_toolkit.experiment_utils.spawn import spawn



def get_cmds_af(args):
    cmds = []
    cmd_i = 0  # index of cmd to-run (but not necessarily launched)
    no_model = 0
    already_computed = 0
    results_dir = args.results_dir#'results_jrdb1'
    cfgs = args.cfgs
    print(f"{cfgs=}")

    if cfgs is None:
        cfgs = []
        assert args.glob_str is not None, "must specify --cfgs or --glob_str"
        for gs in args.glob_str:
            cfgs.extend([str(file).split('.yml')[0].split('/')[-1] for file in Path('cfg').rglob(f'*{gs}*.yml')])

    for cfg in cfgs:
        # check if model exists
        if not args.train:
            if len(glob.glob(f'{results_dir}/{cfg}/*.ckpt')) == 0:
                print("cfg does not have model:", cfg)
                no_model += 1
                if not args.train:
                    continue
            else:
                model_epoch = glob.glob(f'{results_dir}/{cfg}/*=*.ckpt')[-1].split('=')[-1].split('.ckpt')[0]
                # print(f"{cfg} model_epoch:", model_epoch)

        # check if results have already been computed
        if not args.recompute_test and len(glob.glob(f'../trajectory_reward/results/trajectories/test_results/{cfg}.tsv')) > 0:
            print(f"cfg {cfg} results has already been computed")
            already_computed += 1
            continue

        # increment count only after checking if model exists
        cmd_i += 1  # joint-noise running on bacon
        if cmd_i <= args.start_from or len(cmds) >= args.max_cmds:
            continue

        # specify command flags
        if args.train and args.extra_flags is None:
            tail = ' -wb af_jrdb_geom_feats'
        elif args.train and args.extra_flags is not None:
            print("extra_flags:", args.extra_flags)
            tail = " "+args.extra_flags.strip()
        else:
            tail = ' -m test --save_traj'

        cmd = f"python pl_train.py {cfg}{tail}"

        # gather all commands into a list
        cmds.append(cmd)
        # cmds.append('sleep 1; echo "done"')

    print("launching all cmds until cmd_i:", cmd_i)
    assert len(cmds) <= args.max_cmds, f"{len(cmds)} !< {args.max_cmds}"
    print("num no_model:", no_model)
    print("num already_computed:", already_computed)
    print("num yet to compute:", len(cmds))
    return cmds


def get_cmds_vis():
    run_names = ["jrdb_pife_1000_kiss_head_ori",
                  "jrdb_pife_1000_og_odo_head_ori",
                  "jrdb_pife_1000_kiss",
                  "jrdb_pife_1000_og_odo",
                  "jrdb_pife_1000_kiss_blazepose",
                  "jrdb_pife_1000_kiss_head_body_leg_ori",
                  "jrdb_pife_1000_og_odo_blazepose",
                  "jrdb_pife_1000_og_odo_head_body_leg_ori",
                  'jrdb_pife_1000_og_odo_no_kp']

    cmds = []
    for run_name in run_names:
        cmd = f"python jrdb_toolkit/visualisation/visualise2.py --run_name {run_name}"
        cmds.append(cmd)
    return cmds


def get_cmds_vis_compare():

    seqs = [('jrdb_pife_1000_kiss_head_ori','jrdb_pife_1000_kiss_blazepose'),
            ]

    cmds = []
    for run_name0,run_name1 in seqs:
        seq = f'{run_name0}_vs_{run_name1}'
        cmd = f"python jrdb_toolkit/visualisation/visualise2.py --run_name ../results_traj_preds/af_traj_preds/{run_name0} --seq_ids_to_visualize {seq}"
        cmds.append(cmd)
        cmd = f"python jrdb_toolkit/visualisation/visualise2.py --run_name ../results_traj_preds/af_traj_preds/{run_name1} --seq_ids_to_visualize {seq}"
        cmds.append(cmd)
    return cmds


def main(args):
    if args.viz_compare:
        cmds = get_cmds_vis_compare()
    elif args.viz:
        cmds = get_cmds_vis()
    else:
        cmds = get_cmds_af(args)

    spawn(cmds, **vars(args))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_cmds', '-mc', type=int, default=10000)
    argparser.add_argument('--max_cmds_at_a_time', '-c', type=int, default=max(1, multiprocessing.cpu_count()-3))
    argparser.add_argument('--start_from', '-sf', type=int, default=0)
    argparser.add_argument('--train', '-tr', action='store_true')
    argparser.add_argument('--num_gpus_per_node', '-ng', type=int, default=1)
    argparser.add_argument('--cfgs', '-cf', nargs='+', default=None)
    argparser.add_argument('--glob_str', '-g', nargs='+', default=None)
    argparser.add_argument('--viz', '-v', action='store_true')
    argparser.add_argument('--viz_compare', '-vc', action='store_true')
    try:
        cuda_visible = [int(i) for i in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]
    except KeyError:
        cuda_visible = list(range(torch.cuda.device_count()))
    argparser.add_argument('--gpus_available', '-ga', nargs='+', type=int, default=cuda_visible)
    argparser.add_argument('--no_trial', '-nt', dest='trial', action='store_false', help='if not trial, then actually run the commands')
    argparser.add_argument('--redirect_output', '-ro', action='store_true')
    argparser.add_argument('--recompute_test', '-rt', action='store_true')
    argparser.add_argument('--results_dir', '-rd', type=str, default='results_tbd')
    argparser.add_argument('--extra_flags', '-xf', type=str, default=None)

    main(argparser.parse_args())

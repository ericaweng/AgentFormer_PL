"""produce evaluation metrics for two methods and record which one does better"""

import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from eval import eval_one_seq
from jrdb_toolkit.visualisation.visualize_utils import *
from jrdb_toolkit.visualisation.visualize_constants import *
from data.preprocess_w_odometry import get_agents_dict_from_detections, get_agents_features_with_box
from data.preprocess_test_w_odometry import get_agents_features_df_with_box


def get_agents_df(scene):
    """no keypoints"""
    root_location = os.path.expanduser("~/code/AgentFormerSDD/datasets/jrdb")  # path to JRDB

    if scene in TRAIN:
        agents_path = f"{root_location}/processed/"
        agents_dict = get_agents_dict_from_detections(agents_path, scene)
        agents_features = get_agents_features_with_box(agents_dict,
                                                       max_distance_to_robot=MAX_DISTANCE_TO_ROBOT)
        agents_df = pd.DataFrame.from_dict(agents_features, orient='index').rename_axis(['timestep', 'id'])
    else:
        agents_path = f"{root_location}/test/"
        agents_df = get_agents_features_df_with_box(agents_path, TEST_LOCATION_TO_ID[scene],
                                                    max_distance_to_robot=MAX_DISTANCE_TO_ROBOT,
                                                    tracking_method=TRACKING_METHOD)
    return agents_df

def process_sequence(seq_id, preds, scene, method):
    pred_df = pd.DataFrame.from_dict(preds, orient='index').rename_axis(['timestep', 'id'])
    agents_df = get_agents_df(scene)
    trajs_df = pd.merge(pred_df, agents_df, left_index=True, right_index=True, how='inner')

    pred_steps = [seq_id + i * SUBSAMPLE for i in range(1, FUT_STEPS)]
    pred_ids = trajs_df.loc[
        trajs_df.index.get_level_values('timestep').isin(pred_steps) & trajs_df['pred'].apply(
                lambda x: not np.any(np.isnan(x)))].index.get_level_values('id').unique()
    ped_ids_pred = list(map(lambda x: int(x.split(":")[-1]), pred_ids))

    assert 12 * len(pred_ids) == trajs_df.shape[0], f"{12 * len(pred_ids)} != {trajs_df.shape[0]}"

    gt_traj = make_pos_array(trajs_df, pred_steps, ped_ids_pred)[..., :2]
    pred_traj = make_pos_array(trajs_df, pred_steps, ped_ids_pred, 'pred')[..., :2]

    all_metrics, all_ped_vals, all_sample_vals, argmins, collision_mats = eval_one_seq(
            pred_traj[:, None], gt_traj, pred_mask=None, collision_rad=0.1)

    return seq_id, method, all_metrics

def process_method(scene, method):
    pred_traj_path = f'{method}/{scene}'
    pred_trajs_robot_frame = convert_pred_txts_to_dict_global(pred_traj_path)

    method_name = method.split("/")[-1]
    assert len(method_name) > 0

    scene_results = {}
    with ProcessPoolExecutor() as executor:
        futures = [
                executor.submit(process_sequence, seq_id, preds, scene, method_name)
                for seq_id, preds in pred_trajs_robot_frame.items()
        ]

        for future in as_completed(futures):
            seq_id, method_name, all_metrics = future.result()
            scene_results[seq_id] = all_metrics

    return scene, scene_results

def main(args):
    eval_results = {}
    methods = [args.method1, args.method2]

    for scene in tqdm(HST_TEST):
        for method in methods:
            scene, scene_results = process_method(scene, method)
            if scene not in eval_results:
                eval_results[scene] = {}
            eval_results[scene][method] = scene_results

    # aggregate
    method1_better_frames = {}
    ade_comparisons = {}

    # method2_better_frames = {}
    if True:
        differential = 0.5

        for scene in eval_results:
            method1_better_frames[scene] = []
            # method2_better_frames[scene] = []
            ade_comparisons[scene] = {}
            for seq_id, preds in eval_results[scene][args.method1].items():
                m1ade = eval_results[scene][args.method1][seq_id]['ADE_marginal']
                m2ade = eval_results[scene][args.method2][seq_id]['ADE_marginal']
                # if m1ade > m2ade + differential:
                if m1ade < m2ade - differential:
                    method1_better_frames[scene].append(seq_id)
                    ade_comparisons[scene][seq_id]=(m1ade, m2ade)
                # else:
                #     method2_better_frames[scene].append(seq_id)

            method1_better_frames[scene] = sorted(method1_better_frames[scene])
            # method2_better_frames[scene] = sorted(method2_better_frames[scene])

        print(f"{method1_better_frames=}")

    # save to disk
    with open(f'{args.method1.split("/")[-1]}_vs_{args.method2.split("/")[-1]}.txt', 'wb') as f:
        pickle.dump(method1_better_frames, f)
    with open(f'{args.method1.split("/")[-1]}_vs_{args.method2.split("/")[-1]}_ade.txt', 'wb') as f:
        pickle.dump(ade_comparisons, f)
    # with open(f'{args.method2.split("/")[-1]}_vs_{args.method1.split("/")[-1]}.txt', 'wb') as f:
    #     pickle.dump(method2_better_frames, f)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method1', "-m1", type=str, default='')
    parser.add_argument('--method2', "-m2", type=str, default='')
    args = parser.parse_args()

    main(args)
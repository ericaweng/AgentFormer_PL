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
    # (num_preds, num_timesteps, 3)
    pred_traj = make_pos_array(trajs_df, pred_steps, ped_ids_pred, 'pred')[..., :2].swapaxes(1,0)
    # (num_preds, num_samples, num_timesteps, 3)

    all_metrics, all_ped_vals, all_sample_vals, argmins, collision_mats = eval_one_seq(
            pred_traj, gt_traj, pred_mask=None, collision_rad=0.1, return_sample_vals=True)

    return seq_id, method, all_metrics, all_sample_vals


def process_method(scene, method):
    pred_traj_path = f'{method}/{scene}'
    pred_trajs_robot_frame = convert_pred_txts_to_dict(pred_traj_path)

    method_name = method.split("/")[-1]
    if len(method_name) == 0:
        method_name = method.split("/")[-2]

    scene_results = {}
    with ProcessPoolExecutor() as executor:
        futures = [
                executor.submit(process_sequence, seq_id, preds, scene, method_name)
                for seq_id, preds in pred_trajs_robot_frame.items()
        ]

        for future in as_completed(futures):
            seq_id, method_name, all_metrics, min_ade_sample = future.result()
            scene_results[seq_id] = all_metrics | {'best_sample_idx' : np.argmin(min_ade_sample['ADE'])}

    return scene, scene_results


def main(args):
    eval_results = {}
    method = args.method

    for scene in tqdm(HST_TEST[:args.until]):
        scene, scene_results = process_method(scene, method)
        if scene not in eval_results:
            eval_results[scene] = {}
        eval_results[scene] = scene_results

    # aggregate
    notable_frames = {}
    ade_comparisons = {}

    if True:
        threshold = 1.5

        for scene in eval_results:
            notable_frames[scene] = {}
            ade_comparisons[scene] = {}
            for seq_id, preds in eval_results[scene].items():
                ade = eval_results[scene][seq_id]['ADE_marginal']
                if ade > threshold:
                    notable_frames[scene][seq_id]=eval_results[scene][seq_id]['best_sample_idx']
                    ade_comparisons[scene][seq_id]=ade

            # notable_frames[scene] = sorted(notable_frames[scene])

        print(f"{notable_frames=}")

    # save to disk
    with open(f'{args.method.split("/")[-1]}_worst.txt', 'wb') as f:
        pickle.dump(notable_frames, f)
    with open(f'{args.method.split("/")[-1]}_worst_ade.txt', 'wb') as f:
        pickle.dump(ade_comparisons, f)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', "-m", type=str, default='')
    parser.add_argument('--until', "-u", type=int, default=19)
    args = parser.parse_args()

    main(args)


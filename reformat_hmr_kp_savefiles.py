import os
import numpy as np

from data.jrdb_split import get_jrdb_split_full

TRAIN_SEQS, VAL_SEQS, TEST_SEQS = get_jrdb_split_full()
hmr_kp_path = f"datasets/jrdb/jrdb_hmr2_raw_stitched"

for seq_name in TRAIN_SEQS:
    if seq_name in ['bytes-cafe-2019-02-07_0', 'clark-center-2019-02-28_0']:
        continue

    label_3d_pose_path = os.path.join(hmr_kp_path, f"{seq_name}_kp.npz")

    if os.path.exists(os.path.join(hmr_kp_path, f"{seq_name}_kp3d.npz")):
        if os.path.exists(label_3d_pose_path):
            os.remove(label_3d_pose_path)
            print(f"Removed {label_3d_pose_path}")
        print(f"Skipping {seq_name}")
        continue

    print(f"Reformatting {seq_name}")

    with open(label_3d_pose_path, 'rb') as f:
        hmr_labels_3d_pose = np.load(f, allow_pickle=True)['arr_0'].item()
        new_dict = {'pred_cam': {}, 'smpl_params': {}, 'verts': {}, 'kp_2d': {}, 'kp_3d': {}, 'cam_t': {}, 'box_score': {}}
        for frame_id, frames in hmr_labels_3d_pose.items():
            for ped_id, peds in frames.items():
                for key in new_dict.keys():
                    if frame_id not in new_dict[key]:
                        new_dict[key][frame_id] = {}
                new_dict['kp_3d'][frame_id][ped_id] = peds['kp_3d']
                new_dict['smpl_params'][frame_id][ped_id] = peds['smpl_params']
                new_dict['verts'][frame_id][ped_id] = peds['verts']
                new_dict['pred_cam'][frame_id][ped_id] = peds['pred_cam']
                new_dict['cam_t'][frame_id][ped_id] = peds['cam_t']
                new_dict['box_score'][frame_id][ped_id] = peds['box_score']
                new_dict['kp_2d'][frame_id][ped_id] = peds['kp_2d']
        hmr_labels_3d_pose = new_dict

    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_kp2d.npz"), hmr_labels_3d_pose['kp_2d'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_smpl_params.npz"), hmr_labels_3d_pose['smpl_params'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_verts.npz"), hmr_labels_3d_pose['verts'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_pred_cam.npz"), hmr_labels_3d_pose['pred_cam'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_cam_t.npz"), hmr_labels_3d_pose['cam_t'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_box_score.npz"), hmr_labels_3d_pose['box_score'])
    np.savez(os.path.join(hmr_kp_path, f"{seq_name}_kp3d.npz"), hmr_labels_3d_pose['kp_3d'])

    hmr_labels_3d_pose2 = np.load(os.path.join(hmr_kp_path, f"{seq_name}_kp3d.npz"), allow_pickle=True)['arr_0'].item()
    os.remove(label_3d_pose_path)
    print(f"Saved {seq_name} with {len(hmr_labels_3d_pose2)} frames; removed {label_3d_pose_path}")

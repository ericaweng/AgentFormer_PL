"""nvm this file... all agentformer and derivatives, including memonet,
obs.txt peds are in the reverse order as gt.txt, will have to
fix this in the future. todo"""

import glob
import shutil
source = "../trajectory_reward/results/trajectories/sgan/*/*/obs.txt"
for file in glob.glob(source):
    seq_name, frame_name = file.split("/")[-3:-1]
    print("frame_name:", frame_name)
    print("seq_name:", seq_name)
    target = f"../trajectory_reward/results/trajectories/agentformer/{seq_name}/{frame_name}/obs.txt"
    import ipdb; ipdb.set_trace()
    shutil.copyfile(file, target)
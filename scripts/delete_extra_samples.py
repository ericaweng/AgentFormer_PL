"""if there's more samples than 20, delete the extras"""

import glob
import os
from multiprocessing import Pool

# for s in glob.glob('results/zara2_sfm_base*/**/samples/**/frame_*/sample_*.txt', recursive=True):
#     sample_num = s.split('/')[-1].split('.')[0].split("_")[-1]
#     if int(sample_num) > 20:
#         print(s)
#         # import ipdb; ipdb.set_trace()
#         # pass
#         os.remove(s)

files = []
for s in glob.glob('results/zara2_sfm_base*/**/samples/**/frame_*/sample_*.txt', recursive=True):
    sample_num = s.split('/')[-1].split('.')[0].split("_")[-1]
    if int(sample_num) > 20:
        files.append(s)


def rem(f):
    os.remove(f)


with Pool() as p:
    p.map(rem, files)

#Seq crowds_zara02 frame 546 with 7 peds has 46 non-colliding samples in 50 samples
#Seq crowds_zara02 frame 23 with 2 peds has 43 non-colliding samples in 50 samples

ipy train.py --cfg eth_sfm_pre_ --gpu
py test.py --epoch 10 --cfg eth_sfm_pre_ --gpu
py pl_train.py --cfg zara2_1aaat-test --logs_root results-1aaat  --val_every 1 --save_viz --test --dont_resume --test_dataset train --test_ds_size 1 --ckpt_on_test
# testing agentformer 1aaat
ipy pl_train.py --cfg zara2_1aaat-test --logs_root results-1aaat  --val_ever 1 --save_viz --test --dont_resume  --test_ds_size 1

py pl_train.py --cfg zara1_agentformer_pre  -dz 5 -v -d test -m test  --test -c last -nmp -g
ipy pl_train.py --cfg zara1_agentformer_pre  -dz 5  --test -nc -l


# joint baseline evaluations
ipy pl_train.py --cfg zara2_agentformer_nocol --logs_root results-joint  -m test  --save_traj


# 2 agent af control (plain AF)
python3 -m ipdb pl_train.py --cfg zara2-1aaat-pre-2 --logs_root res-new --val_every 1 --save_viz --dont_resume --test
python3 -m ipdb pl_train.py --cfg zara2-1aaat-pre-2 --logs_root res-new -m test -c res-new/zara2-1aaat-pre-2/
# 2 agent af 1aaat
python3 -m ipdb pl_train.py --cfg zara2_1aaat-test-2 --logs_root res-new --val_every 1 --save_viz --dont_resume --test
python3 -m ipdb pl_train.py --cfg zara2_1aaat-test-2 --logs_root res-new --save_viz --dont_resume -m test -c res-new/zara2_1aaat-test-2/
# 1 agent af control (plain AF)
python3 -m ipdb pl_train.py --cfg zara2_1aaat --logs_root res-new --val_every 1 --save_viz --dont_resume
python3 -m ipdb pl_train.py --cfg zara2_1aaat --logs_root res-new -m test -c res-new/zara2_1aaat/
# 1 agent af 1aaat
python3 -m ipdb pl_train.py --cfg zara2_1aaat-test --logs_root res-new --val_every 1 --save_viz --dont_resume
python3 -m ipdb pl_train.py --cfg zara2_1aaat-test --logs_root res-new -m test -c res-new/zara2_1aaat-test/

# Notes
# - LR 1e-3 too high; must be 5e-4 or lower
# -
ipy train.py --cfg eth_sfm_pre_ --gpu
py test.py --epoch 10 --cfg eth_sfm_pre_ --gpu
py pl_train.py --cfg zara2_1aaat-test --logs_root results-1aaat  --val_every 1 --save_viz --test --dont_resume --test_dataset train --test_ds_size 1 --ckpt_on_test
# testing agentformer 1aaat
ipy pl_train.py --cfg zara2_1aaat-test --logs_root results-1aaat  --val_ever 1 --save_viz --test --dont_resume  --test_ds_size 1
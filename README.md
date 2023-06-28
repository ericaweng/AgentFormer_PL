pytorch lightning implementation of [Agentformer](https://github.com/Khrylx/AgentFormer)

main file: `pl_train.py`

pytorch lightning Trainer file: `trainer.py`

to train Joint AgentFormer (AgentFormer with joint ADE loss):
```
python pl_train.py --cfg <dset>_af_pre_mg-1-jr-1
```
where `<dset>` is one of `eth`, `hotel`, `univ`, `zara1`, `zara2`.
After that finishes training, train the DLow model. The DLow model has improved diversity predictions per sequence (more diverse modes per agent).
```
python pl_train.py --cfg <dset>_af_mg-1,5_jr-1,5
```

here are all available flags:
```
--cfg: name of the config file to run
--mode: either "train" "test" or "val"
--batch_size: only batch size 1 is available right now, sorry :-(
--no_gpu: specify if you want CPU-only training
--dont_resume: specify if you don't want to resume from checkpoint if it exists
--checkpoint_path: specify if you want to resume from a model different than the default (which is ./results-joint/<args.cfg>)
--save_viz: save visualizations to ./viz
--save_num: num  to visualizations save per eval step
--logs_root: default root dir to save logs and model checkpoints. default is ./results-joint and logs for a run will be saved to <args.logs_root>/<args. cfg>
--save_traj: whether to save trajectories for offline evaluation
```

The code is adapted for pytorch lightning (multi-gpu training) from:
AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting  
Ye Yuan, Xinshuo Weng, Yanglan Ou, Kris Kitani  
**ICCV 2021**  
[[website](https://www.ye-yuan.com/agentformer)] [[paper](https://arxiv.org/abs/2103.14023)]

if you find this code useful, we would appreciate if you cite:

```
@misc{weng2023joint,
      title={Joint Metrics Matter: A Better Standard for Trajectory Forecasting}, 
      author={Erica Weng and Hana Hoshino and Deva Ramanan and Kris Kitani},
      year={2023},
      eprint={2305.06292},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

```
@misc{yuan2021agentformer,
      title={AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting}, 
      author={Ye Yuan and Xinshuo Weng and Yanglan Ou and Kris Kitani},
      year={2021},
      eprint={2103.14023},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


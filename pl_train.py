import os
import sys
import glob
import argparse
from functools import partial

import torch
import wandb
torch.set_default_dtype(torch.float32)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import AgentFormerDataModule
from utils.config import Config
from utils.utils import get_timestring
from trainer import AgentFormerTrainer


def main(args):
    # initialize AgentFormer config from wandb sweep config or from full config filename
    if args.wandb_sweep:
        run = wandb.init()
        run.name = full_config_name = f'{args.cfg}_w-{wandb.config.weight:0.1f}_s-{wandb.config.sigma_d:0.2f}'
        cfg = Config(full_config_name)
        wandb.config.update(args)
        wandb.config.update(cfg)
    else:
        cfg = Config(args.cfg)

    # Set global random seed
    pl.seed_everything(cfg.seed)

    # run with test run params if in ipython env; else with real run params
    try:
        __IPYTHON__
        in_ipdb = True
        print("TEST RUN: in ipdb")
    except NameError:  # no ipdb; real run
        in_ipdb = False
        print("not in ipdb")

    if args.no_gpu:
        accelerator = None
        plugin = None
    else:
        if not in_ipdb and not args.mode == 'test':
            plugin = DDPStrategy(find_unused_parameters=args.find_unused_params)
            args.devices = torch.cuda.device_count()
            accelerator = 'gpu'
        else:
            plugin = None
            args.devices = 1
            accelerator = 'gpu'

    print(f"using {args.devices} gpus")
    if args.test:
        sanity_val_steps = 0
        lim_train_batch = int(args.test_ds_size / args.batch_size)
        lim_val_batch = int(args.test_ds_size / args.batch_size)
    else:
        sanity_val_steps = 1
        lim_train_batch = None
        lim_val_batch = None

    # load checkpoint model
    default_root_dir = args.default_root_dir = os.path.join(args.logs_root, cfg.id)
    if args.mode == 'train':
        models = sorted(glob.glob(os.path.join(default_root_dir, 'last-v*.ckpt')))
        if len(models) == 0:
            models = sorted(glob.glob(os.path.join(default_root_dir, 'last*.ckpt')))
    elif args.mode == 'test' or args.mode == 'val':
        if args.checkpoint_str is not None:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*{args.checkpoint_str}*.ckpt')))
        else:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*epoch=*.ckpt')))
    else:
        raise NotImplementedError
    print("models:", models)

    if args.checkpoint_path is not None and args.resume:
        resume_from_checkpoint = args.checkpoint_path
        print("LOADING from custom checkpoint:", resume_from_checkpoint)
    elif len(models) > 0 and os.path.isfile(models[-1]) and args.resume:
        resume_from_checkpoint = models[-1]
        print("LOADING from default model directory:", resume_from_checkpoint)
    else:
        resume_from_checkpoint = None
        print("STARTING new run from scratch")

    # initialize DataModule and Trainer
    dm = AgentFormerDataModule(cfg, args)
    model = AgentFormerTrainer(cfg, args)
    if args.log_graph:
        model.set_example_input_array(next(iter(dm.train_dataloader())))
    model.model.set_device(model.device)

    # initialize logging and checkpointing and other training utils
    if args.mode == 'train' and (not args.test or args.log_on_test):
        logger = TensorBoardLogger(args.logs_root, name=cfg.id, log_graph=args.log_graph)
        if args.wandb_sweep:
            wandb_logger = WandbLogger(save_dir=args.logs_root, log_model=True, save_top_k=5, name=run.name)
            logger = [logger, wandb_logger]
    else:
        logger = None
    early_stop_cb = EarlyStopping(patience=20, verbose=True, monitor='val/ADE_joint')
    checkpoint_callback = ModelCheckpoint(monitor='val/ADE_joint', save_top_k=5, mode='min', save_last=True,
                                          every_n_epochs=1, dirpath=default_root_dir, filename='{epoch:04d}')
    tqdm = TQDMProgressBar(refresh_rate=args.tqdm_rate)

    callbacks = [tqdm]
    if args.test and args.ckpt_on_test or not args.test:
        callbacks.append(checkpoint_callback)
    if not args.test:
        callbacks.append(early_stop_cb)

    print("LOGGING TO:", default_root_dir)
    print("\n\n")
    trainer = pl.Trainer(check_val_every_n_epoch=args.val_every, num_sanity_val_steps=sanity_val_steps,
                         devices=args.devices, strategy=plugin, accelerator=accelerator,
                         log_every_n_steps=50 if lim_train_batch is None else lim_train_batch,
                         limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                         max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                         logger=logger, callbacks=callbacks,)

    if 'train' in args.mode:
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
        trainer = pl.Trainer(devices=1, accelerator=accelerator, default_root_dir=default_root_dir)
        args.mode = 'test'
        args.save_traj = True
        args.save_viz = True
        dm = AgentFormerDataModule(cfg, args)
        model = AgentFormerTrainer(cfg, args)
        model.model.set_device(model.device)
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    elif 'test' in args.mode or 'val' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_sfm_pre8-2')
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--no_gpu', '-ng', action='store_true', default=False)
    parser.add_argument('--dont_resume', '-dr', '-nc', dest='resume', action='store_false', default=True)
    parser.add_argument('--checkpoint_path', '-cp', default=None)
    parser.add_argument('--checkpoint_str', '-c', default=None)
    parser.add_argument('--test', '-t', action='store_true', default=False)
    parser.add_argument('--no_mp', '-nmp', dest='mp', action='store_false', default=True)
    parser.add_argument('--save_viz', '-v', action='store_true', default=False)
    parser.add_argument('--save_num', '-vn', type=int, default=10, help='number of visualizations to save per eval')
    parser.add_argument('--logs_root', '-lr', default='results-joint', help='where to save checkpoints and tb logs')
    parser.add_argument('--log_on_test', '-l', action='store_true', default=False, help='if true, then also writes logs when --test is also specified (o/w does not)')
    parser.add_argument('--ckpt_on_test', '-ck', action='store_true', default=False)
    parser.add_argument('--save_traj', '-s', action='store_true', default=False)
    parser.add_argument('--log_graph', '-g', action='store_true', default=False)
    parser.add_argument('--find_unused_params', '-f', action='store_true', default=False)
    parser.add_argument('--tqdm_rate', '-tq', type=int, default=20)
    parser.add_argument('--val_every', '-ve', type=int, default=5)
    parser.add_argument('--test_ds_size', '-dz', default=10, type=int, help='max size of dataset to load when using the --test flag')
    parser.add_argument('--test_dataset', '-d', default='test', help='which dataset to test on (train for sanity-checking)')
    parser.add_argument('--frames_list', '-fl', default=None, type=lambda x: list(map(int, x.split(','))), help='test only certain frame numbers')
    parser.add_argument('--start_frame', '-sf', default=None, type=int, help="frame to start loading data from, if you don't want to load entire dataset")
    parser.add_argument('--dont_save_test_results', '-dstr', dest='save_test_results', action='store_false', default=True, help='whether or not to save test results stats to tsv file (on test mode)')
    parser.add_argument('--wandb_sweep', '-ws', action='store_true', default=False, help='runs wandb sweep with id given by args.sweep_id')
    parser.add_argument('--sweep_id', '-sid', default=None, help='if given, continues wandb sweep with id given by args.sweep_id')
    parser.add_argument('--new_sweep', '-ns', action='store_true', default=False, help='if true, then starts new wandb sweep')
    parser.add_argument('--project_name', '-p', default='af_dlow_sfm', help='wandb project name')
    args = parser.parse_args()

    time_str = get_timestring()
    print(f"time str: {time_str}")
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print(f"torch version : {torch.__version__}")
    print(f"cudnn version : {torch.backends.cudnn.version()}")

    if args.wandb_sweep:
        configs = {
                'weight': None,
                'sigma_d': None,
        }
        sweep_config = {
            'method': 'grid',
            'metric': {
                'goal': 'minimize',
                'name': 'val/ADE_joint'
            },
            'parameters': {
                'weight': {
                    'values': [10, 15, 20, 30, 40, 50, 100]
                },
                'sigma_d': {
                    'values': [0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5]
                }
            }
        }
        # get sweep id from file if not given as arg
        if args.sweep_id is None and not args.new_sweep:
            try:
                with open('sweep_id2.txt', 'r') as f:
                    args.sweep_id = f.read().strip()
            except:
                args.sweep_id = wandb.sweep(sweep_config, project=args.project_name)
                with open('sweep_id2.txt', 'w') as f:
                    f.write(args.sweep_id)
        else:
            args.sweep_id = wandb.sweep(sweep_config, project=args.project_name)
            with open('sweep_id2.txt', 'w') as f:
                f.write(args.sweep_id)
        wandb.agent(args.sweep_id, function=partial(main, args), project=args.project_name)
        print(f"sweep id: {args.sweep_id}")
    else:
        main(args)

import os
import glob
import argparse
import numpy as np
import torch
# torch.set_float32_matmul_precision('high')
# torch.set_default_dtype(torch.float32)
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy

from data.datamodule import AgentFormerDataModule
from utils.config import Config
from trainer import AgentFormerTrainer
from callbacks import ModelCheckpointCustom


def main(args):
    # initialize AgentFormer config from full config filename
    cfg = Config(args.cfg)

    # Set global random seed
    pl.seed_everything(cfg.seed)
    print("Setting global seed:", cfg.seed)

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
        plugin = 'ddp'
    else:
        if not in_ipdb and not args.mode == 'test':
            plugin = DDPStrategy(find_unused_parameters=args.find_unused_params)
            args.devices = torch.cuda.device_count()
            accelerator = 'gpu'
        else:
            plugin = None#'ddp_notebook'
            args.devices = 1
            accelerator = 'gpu'

    print(f"using {args.devices} gpus")
    if args.trial:
        sanity_val_steps = 0
        lim_train_batch = int(args.trial_ds_size / args.batch_size)
        lim_val_batch = int(args.trial_ds_size / args.batch_size)

        # stuff
        args.ckpt_on_trial = True
        args.val_every = 1
        args.mp = False
        args.save_viz_every_time = True
        args.save_viz = True
    else:
        sanity_val_steps = 1
        lim_train_batch = None
        lim_val_batch = None

    # load checkpoint model
    default_root_dir = args.default_root_dir = os.path.join(args.logs_root, cfg.id)
    models_ronnie = []
    if args.mode == 'train':
        # specify checkpoint to resume for dlow
        resume_cfg = cfg.get('resume_cfg', None)
        if resume_cfg is not None:
            models = sorted(glob.glob(os.path.join(args.logs_root, resume_cfg, '*epoch=*.ckpt')))
            print("tuning via resuming from best checkpoint of other model")
        elif args.checkpoint_str is not None:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*{args.checkpoint_str}*.ckpt')))
        else:  # resume from last checkpoint
            models = sorted(glob.glob(os.path.join(default_root_dir, 'last-v*.ckpt')))
            if len(models) == 0:
                models = sorted(glob.glob(os.path.join(default_root_dir, 'last*.ckpt')))
                if len(models) == 0:  # load ronnie version a checkpoint and adjust so that it's compatible
                    models = sorted(glob.glob(os.path.join(default_root_dir, '*epoch=*.ckpt')))
                    # models_ronnie = sorted(glob.glob(os.path.join(default_root_dir, '*.p')))
    elif args.mode == 'test' or args.mode == 'val':
        if args.checkpoint_str is not None:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*{args.checkpoint_str}*.ckpt')))
        else:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*epoch=*.ckpt')))
        if len(models) == 0:
            raise FileNotFoundError(f'If testing, must exist model in {default_root_dir}')
    else:
        models = []
    print("models:", models)

    if args.checkpoint_path is not None and args.resume:
        resume_from_checkpoint = args.checkpoint_path
        print("LOADING from custom checkpoint:", resume_from_checkpoint)
        print("LOADING from default model directory:", resume_from_checkpoint)
        args.current_epoch_model = resume_from_checkpoint.split('.ckpt')[0].split('epoch=')[-1]
    elif len(models) > 0 and os.path.isfile(models[-1]) and args.resume:
        resume_from_checkpoint = models[-1]
        print("LOADING from default model directory:", resume_from_checkpoint)
        # load model with torch
        checkpoint = torch.load(resume_from_checkpoint)
        # print(f"checkpoint: {checkpoint.keys()}")
        if len(models_ronnie) > 0:
            ckpt_ronnie = torch.load(models_ronnie[-1])
            print(f"checkpoint ronnie: {ckpt_ronnie.keys()}")
            # replace the model weights with the ronnie model and resave as epoch={epoch}.ckpt
            ckpt_ronnie['pytorch-lightning_version'] = checkpoint['pytorch-lightning_version']
            ronnie_epoch = models_ronnie[-1].split('_')[-1].split('.')[0]
            # checkpoint['state_dict'] = ckpt_ronnie['state_dict']
            # checkpoint['global_step'] = None
            # checkpoint['loops'] = None
            # checkpoint['callbacks'] = None
            path_to_save = os.path.join(default_root_dir, f'epoch={ronnie_epoch}.ckpt')
            torch.save(ckpt_ronnie, path_to_save)
            # delete ronnie model
            os.remove(models_ronnie[-1])
            resume_from_checkpoint = path_to_save
        args.current_epoch_model = resume_from_checkpoint.split('.ckpt')[0].split('epoch=')[-1]
    else:
        resume_from_checkpoint = None
        print("STARTING new run from scratch")
        args.current_epoch_model = None

    # initialize DataModule and Trainer
    dm = AgentFormerDataModule(cfg, args)
    model = AgentFormerTrainer(cfg, args)
    model.model.set_device(model.device)

    # initialize logging and checkpointing and other training utils
    if args.mode == 'train' and (not args.trial or args.log_on_trial):
        if args.wandb_project_name is not None:
            suffix = f' ({args.tag})' if args.tag is not None else ''
            logger = WandbLogger(name=f'{cfg.id}{suffix}', save_dir=args.logs_root, project=args.wandb_project_name,
                                 config={**vars(cfg), 'args': vars(args)})
        else:
            logger = TensorBoardLogger(args.logs_root, name=cfg.id)
    else:
        logger = None
    early_stop_cb = EarlyStopping(patience=5, verbose=True, monitor='val/ADE_joint')
    checkpoint_callback = ModelCheckpointCustom(visualize=args.save_viz, monitor='val/ADE_joint', mode='min',
                                                save_last=True, save_top_k=5, dirpath=default_root_dir,
                                                filename='{epoch:04d}',)
    tqdm = TQDMProgressBar(refresh_rate=args.tqdm_rate)

    callbacks = [tqdm]
    if args.trial and args.ckpt_on_trial or not args.trial:
        callbacks.append(checkpoint_callback)
    if not args.trial:
        callbacks.append(early_stop_cb)

    print("LOGGING TO:", default_root_dir)
    print("\n\n")

    if 'train' in args.mode:
        trainer = pl.Trainer(check_val_every_n_epoch=args.val_every, num_sanity_val_steps=sanity_val_steps,
                             devices=args.devices, strategy=plugin, accelerator=accelerator,
                             log_every_n_steps=50 if lim_train_batch is None else lim_train_batch,
                             limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                             max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                             logger=logger,
                             callbacks=callbacks, )  # deterministic=True)  # not needed to control for proper training batch order

        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
        trainer = pl.Trainer(devices=1, accelerator=accelerator, default_root_dir=default_root_dir, logger=None,# logger=logger,  # don't log to tb or wandb on test
                             deterministic=True)
        args.save_viz = True
        model.update_args(args)
        trainer.test(model, datamodule=dm)
    elif 'test' in args.mode or 'val' in args.mode:
        trainer = pl.Trainer(devices=1, accelerator=accelerator, default_root_dir=default_root_dir, logger=None,# logger=logger,
                             deterministic=True)
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    elif 'check_dl' in args.mode:
        cfg.dataloader_version = 0
        dl_train0 = dm.get_dataloader('train')
        # dl_test0 = dm.get_dataloader('test')
        cfg.dataloader_version = 2
        dl_train2 = dm.get_dataloader('train')
        # dl_test = dm.get_dataloader('test')

        # check that the two dataloaders are the same
        from check_dataloader import process_data_in_parallel
        process_data_in_parallel(dl_train0, dl_train2, args.num_workers)
        # process_data_in_parallel(dl_test0, dl_test, args.num_workers)

    elif 'viz' in args.mode:
        print("visualizing ground truth")
        from viz_utils_plot import _save_viz_gt

        num_cpus = args.num_workers
        to_save_data = []
        for i, data in enumerate(dm.get_dataloader('train')):
            to_save_data.append({'gt_motion': np.stack(data['fut_motion'], 1),
            'obs_motion': np.stack(data[f'pre_motion'], 1),
            'frame': data['frame'],
            'seq': data['seq'],
            'data': {'heading_avg': data['heading_avg'], 'heading': data['heading']}
            })
            if len(to_save_data) >= num_cpus:
                _save_viz_gt(to_save_data, args, 'train')
                to_save_data = []
        if len(to_save_data) > 0:
            _save_viz_gt(to_save_data, args, 'train')
            to_save_data = []
        for i, data in enumerate(dm.get_dataloader('val')):
            val_data = {'gt_motion': np.stack(data['fut_motion'], 1),
            'obs_motion': np.stack(data[f'pre_motion'], 1),
            'frame': data['frame'],
            'seq': data['seq'],
            'data': {'heading_avg': data['heading_avg'], 'heading': data['heading']}
            }
            to_save_data.append(val_data)
            if len(to_save_data) >= num_cpus:
                _save_viz_gt(to_save_data, args, 'val')
                to_save_data = []
        if len(to_save_data) > 0:
            _save_viz_gt(to_save_data, args, 'val')
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg')
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', '-nw', type=int, default=24)
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--no_gpu', '-ng', action='store_true', default=False)
    parser.add_argument('--dont_resume', '-dr', '-nc', dest='resume', action='store_false', default=True)
    parser.add_argument('--checkpoint_path', '-cp', default=None)
    parser.add_argument('--checkpoint_str', '-c', default=None)
    parser.add_argument('--trial', '-t', action='store_true', default=False, help='if true, then does a trial run (without save checkpoints or logs, and allows user to specify smaller dataset size for sanity checking)')
    parser.add_argument('--no_mp', '-nmp', dest='mp', action='store_false', default=True)
    parser.add_argument('--no_viz', '-nv', dest='save_viz', action='store_false')
    parser.add_argument('--save_viz_every_time', '-vv', action='store_true', default=False)
    parser.add_argument('--save_num', '-vn', type=int, default=10, help='number of visualizations to save per eval')
    parser.add_argument('--logs_root', '-lr', default='results_jrdb1', help='where to save checkpoints and tb logs')
    parser.add_argument('--log_on_trial', '-lot', action='store_true', default=False)
    parser.add_argument('--ckpt_on_trial', '-l', '-ck', action='store_true', default=False)
    parser.add_argument('--save_traj', '-s', action='store_true', default=False)
    parser.add_argument('--find_unused_params', '-f', action='store_true', default=False)
    parser.add_argument('--tqdm_rate', '-tq', type=int, default=20)
    parser.add_argument('--val_every', '-ve', type=int, default=3)
    parser.add_argument('--trial_ds_size', '-dz', default=10, type=int, help='max number of scenes to load when using the --trial flag')
    parser.add_argument('--randomize_trial_data', '-rtd', action='store_true', default=False)
    parser.add_argument('--test_dataset', '-d', default='test', help='which dataset to test on (train for sanity-checking)')
    parser.add_argument('--frames_list', '-fl', default=None, type=lambda x: list(map(int, x.split(','))), help='test only certain frame numbers')
    parser.add_argument('--start_frame', '-sf', default=None, type=int, help="frame to start loading data from, if you don't want to load entire dataset")
    parser.add_argument('--dont_save_test_results', '-dstr', dest='save_test_results', action='store_false', default=True, help='whether or not to save test results stats to tsv file (on test mode)')
    parser.add_argument('--wandb_project_name', '-wb', default='jrdb')#None, help='wandb project name')
    parser.add_argument('--interaction_category', '-ic', action='store_true')#default='all', help='which interaction category to test on')
    parser.add_argument('--seq_frame', '-sqf', default=None)
    parser.add_argument('--tag', '-tg', default=None)

    args = parser.parse_args()

    main(args)

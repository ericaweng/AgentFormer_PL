import os
import sys
import glob
import argparse

import torch
torch.set_default_dtype(torch.float32)

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import AgentFormerDataModule
from utils.config import Config
from utils.utils import get_timestring
from trainer import AgentFormerTrainer


def main(args):
    # Set global random seed
    cfg = Config(args.cfg)
    pl.seed_everything(cfg.seed)

    # run with test run params if in ipython env; else with real run params
    try:
        __IPYTHON__
        in_ipdb = True
        print("TEST RUN: in ipdb")
    except NameError:  # no ipdb; real run
        in_ipdb = False
        print("not in ipdb")

    if not in_ipdb and not args.mode == 'test':
        plugin = DDPStrategy(find_unused_parameters=args.find_unused_params)
        args.devices = torch.cuda.device_count()
        accelerator = 'gpu'
    else:
        plugin = None
        args.devices = 1
        # devices = None
        # accelerator = None
        accelerator = 'gpu'

    if args.test:
        sanity_val_steps = 0
        lim_train_batch = 5
        lim_val_batch = 5
    else:
        sanity_val_steps = 1
        lim_train_batch = None
        lim_val_batch = None

    # Initialize data module
    dm = AgentFormerDataModule(cfg, args)
    if args.log_graph:
        example_input_array = [next(iter(dm.train_dataloader()))]
    else:
        example_input_array = None
    model = AgentFormerTrainer(cfg, args, example_input_array)
    model.model.set_device(model.device)

    # Initialize trainer
    default_root_dir = os.path.join(args.logs_root, cfg.id)
    if args.mode == 'train':
        models = sorted(glob.glob(os.path.join(default_root_dir, 'last-v*.ckpt')))
        if len(models) == 0:
            models = sorted(glob.glob(os.path.join(default_root_dir, 'last*.ckpt')))
    elif args.mode == 'test' or args.mode == 'val':
        if args.checkpoint_str is not None:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*{args.checkpoint_str}*.ckpt')))
        else:
            models = sorted(glob.glob(os.path.join(default_root_dir, f'*epoch=*{args.checkpoint_str}*.ckpt')))
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
    # Initialize trainer
    if args.mode == 'train' and (not args.test or args.log_on_test):
        logger = TensorBoardLogger(args.logs_root, name=cfg.id, log_graph=args.log_graph)
    else:
        logger = None
    early_stop_cb = EarlyStopping(patience=20, verbose=True, monitor='val/ADE')
    checkpoint_callback = ModelCheckpoint(monitor='val/ADE', save_top_k=3, mode='min', save_last=True,
                                          every_n_epochs=1, dirpath=default_root_dir, filename='{epoch:04d}')

    print("LOGGING TO:", default_root_dir)
    print("\n\n")
    trainer = pl.Trainer(check_val_every_n_epoch=5, num_sanity_val_steps=sanity_val_steps,
                         devices=args.devices, strategy=plugin, accelerator=accelerator,
                         log_every_n_steps=50 if lim_train_batch is None else lim_train_batch,
                         limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                         max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                         logger=logger, callbacks=[early_stop_cb, checkpoint_callback],)

    if 'train' in args.mode:
        n = trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
        if n:
            trainer = pl.Trainer(devices=1, accelerator=accelerator, default_root_dir=default_root_dir)
            trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    elif 'test' in args.mode or 'val' in args.mode or 'cond' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_sfm_pre8-2')
    parser.add_argument('--mode', '-m', default='train')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--dont_resume', '-dr', '-nc', dest='resume', action='store_false', default=True)
    parser.add_argument('--checkpoint_path', '-cp', default=None)
    parser.add_argument('--checkpoint_str', '-c', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--no_mp', '-nmp', dest='mp', action='store_false', default=True)
    parser.add_argument('--save_viz', '-v', action='store_true', default=False)
    parser.add_argument('--logs_root', default='results2')
    parser.add_argument('--log_on_test', '-l', action='store_true', default=False)
    parser.add_argument('--log_graph', '-g', action='store_true', default=False)
    parser.add_argument('--find_unused_params', '-f', action='store_true', default=False)
    parser.add_argument('--test_ds_size', '-t', default=10, type=int,
                        help='max size of dataset to load when using the --test flag')
    parser.add_argument('--test_dataset', '-d', default='test', help='which dataset to test on (train for sanity-checking)')
    args = parser.parse_args()

    time_str = get_timestring()
    print("time str: {}".format(time_str))
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print("torch version : {}".format(torch.__version__))
    print("cudnn version : {}".format(torch.backends.cudnn.version()))
    main(args)

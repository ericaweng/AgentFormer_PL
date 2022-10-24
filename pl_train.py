import os
import sys
import argparse

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from data.datamodule import AgentFormerDataModule
from utils.config import Config
from utils.utils import get_timestring
from trainer import AgentFormerTrainer
from torch import optim


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

    if not in_ipdb:
        plugin = DDPStrategy(find_unused_parameters=False)
        num_gpus = torch.cuda.device_count()
    else:
        plugin = None
        num_gpus = 0

    if args.test:
        sanity_val_steps = 0
        lim_train_batch = 20
        lim_val_batch = 20
    else:
        sanity_val_steps = 1
        lim_train_batch = None
        lim_val_batch = None

    torch.set_default_dtype(torch.float32)

    # Initialize data module
    dm = AgentFormerDataModule(cfg, args)
    model = AgentFormerTrainer(cfg)

    # Initialize trainer
    last_model_path = os.path.join(cfg.model_dir, 'pl_model_last.p')
    if args.checkpoint_path is not None:
        resume_from_checkpoint = args.checkpoint_path
        print("LOADING from custom checkpoint:", resume_from_checkpoint)
    elif os.path.isfile(last_model_path):
        resume_from_checkpoint = last_model_path
        print("LOADING from default model directory specified by the cfg:", resume_from_checkpoint)
    else:
        resume_from_checkpoint = None
    # Initialize trainer
    logger = TensorBoardLogger(args.logs_root, name=cfg.id)#cfg.result_dir, name='tb')#, version=args.experiment_name)
    early_stop_cb = EarlyStopping(patience=20, verbose=True, monitor='val/ADE')
    checkpoint_callback = ModelCheckpoint(monitor='val/ADE', save_top_k=3, mode='min', save_last=True,
                                          every_n_epochs=1, dirpath=cfg.model_dir, filename='model_{epoch:04d}.p')

    default_root_dir = os.path.join(args.logs_root, cfg.id)
    print("LOGGING TO:", default_root_dir)
    print("\n\n")
    trainer = pl.Trainer(check_val_every_n_epoch=cfg.print_freq, num_sanity_val_steps=sanity_val_steps,
                         gpus=num_gpus, strategy=plugin,
                         limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                         max_epochs=cfg.num_epochs, default_root_dir=default_root_dir,
                         logger=logger, callbacks=[early_stop_cb, checkpoint_callback],)

    if 'train' in args.mode:
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
    elif 'test' in args.mode or 'val' in args.mode or 'cond' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_sfm_pre8-2')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--resume_from_checkpoint', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', default=None)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--logs_root', default='results2')
    args = parser.parse_args()

    time_str = get_timestring()
    print("time str: {}".format(time_str))
    print("python version : {}".format(sys.version.replace('\n', ' ')))
    print("torch version : {}".format(torch.__version__))
    print("cudnn version : {}".format(torch.backends.cudnn.version()))
    main(args)

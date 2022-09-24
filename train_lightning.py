import json
import os
import torch
import pytorch_lightning as pl
import sys

from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin


def parse_arguments():
    parser = ArgumentParser()

    # Load experiment and trainer-sepcific args
    parser = add_experimental_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)

    # Load dataset specific args
    parser = ArgoverseDataModule.add_data_specific_args(parser)

    # Load model specific args
    parser = get_args(parser.parse_known_args()[0].model_name, parser)

    args = parser.parse_args()


    return args


def add_experimental_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # Parse dataset model to use
    parser.add_argument('--dataset', type=str, default='ArgoverseDatasetLaneGraph', help='Name of dataset to use')
    parser.add_argument('--model-name', type=str, required=True, help='Name of model to load')

    # General Params
    parser.add_argument("--mode", required=True, type=str,
                        # choices=['train', 'val', 'trainval', 'oracle-val', 'test', 'argoverse-test', 'sanity', 'sanity-test'],
                        help='Mode to run forecasting model in')
    parser.add_argument('--seed', type=int, help="Seed the random parameter generation")
    parser.add_argument('--no-batch-norm', dest='batch_norm', action='store_false',
                        help="Seed the random parameter generation")
    parser.add_argument("--viz", action='store_true', help='Visualize predictions')
    parser.add_argument("--test", action='store_true', help="test mode: doesn't shuffle train dataset")
    parser.add_argument("--val", action='store_true',
                        help="val mode: train without training doesn't shuffle train dataset")
    parser.add_argument("--plot-all-tracks", action='store_true', help='plot social actor tracks')
    parser.add_argument("--write_metrics_to_file", action='store_true',
                        help='write metrics of each scenario (ade / fde) to file')
    parser.add_argument("--comment", help='comment appended to end of visualizations filename')
    parser.add_argument("--path_to_interesting_scenario_list", default='', help='')

    # I/O and Feature Space Params
    parser.add_argument('--delta', action='store_true', help="Predict delta-xy coordinates instead of absolute")  # NOQA
    parser.add_argument('--training-signal', choices=['cond', 'focal', 'all'], default='cond',
                        help="focal: train only on focal agent's loss\ncond: train on social agents loss "
                             "(and condition on focal agent future GT)\nall: train on focal and social agents loss")
    parser.add_argument('--cond-all', action='store_true',
                        help="if true, will redraw training_signal every training step")
    parser.add_argument("--train-noise", action='store_true', help='Use rotational noise during training')
    parser.add_argument('--num-futures', type=int, default=6, help='Number of possible futures')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of gpus to use')

    # Datamodule Params
    parser.add_argument('--dataroot', required=True, help="Path to the processed dataset folder")
    parser.add_argument("--batch-size", type=int, default=25, help="Training batch size")
    parser.add_argument('--workers', type=int, default=8, help="Number of dataloader workers")

    # Trainer Params
    parser.add_argument("--gpus", type=int, default=1, help='# of GPUs to use for training')
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1,
                        help="# of training epochs between val")  # NOQA
    parser.add_argument("--max-epochs", type=int, default=150, help="Max # of training epochs")
    parser.add_argument("--early-stop-threshold", type=int, default=20,
                        help="Number of consecutive val epochs without improvement before termination")  # NOQA
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes used')
    parser.add_argument('--precision', default=32, type=int, help='Precision employed in weights')
    parser.add_argument('--resume-from-checkpoint', default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--distributed-backend', default='ddp', help='Trainer backend')
    parser.add_argument("--timesteps", type=int, default=50, help="Number of input timesteps")
    parser.add_argument("--outsteps", type=int, default=60, help="Number of output timesteps")

    # Logging Params
    parser.add_argument('--resume-from-checkpoint-legacy', default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(os.getcwd(), 'models'),
                        help='Path to save files')

    # Logging Params
    parser.add_argument('--experiment-name', type=str, help='Save file prefix')
    parser.add_argument('--logs-root', type=str, default=os.path.join(os.getcwd(), 'models') + '/experiments',
                        help='Path to save logs')

    return parser


def cli_main(args):
    # print(args)

    # Set global random seed
    pl.seed_everything(args.seed)

    # run with test run params if in ipython env; else with real run params
    try:
        __IPYTHON__
        in_ipdb = True
        print("TEST RUN: in ipdb")
    except NameError:  # no ipdb; real run
        in_ipdb = False
        print("not in ipdb")

    if not in_ipdb:
        plugins = DDPPlugin(find_unused_parameters=False)
        args.num_gpus = torch.cuda.device_count() if args.num_gpus is None else args.num_gpus
    else:
        plugins = []
        args.num_gpus = 0

    if args.test:
        sanity_val_steps = 0
        lim_train_batch = 1
        lim_val_batch = 1
        args.max_epochs = 10000
    else:
        lim_train_batch = None
        lim_val_batch = None
        sanity_val_steps = 1

    plugins = []
    args.num_gpus = 1

    # Initialize data module
    dm = ArgoverseDataModule(args)
    model = get_model(args.model_name, args)

    # Initialize trainer
    resume_from_checkpoint = None
    if os.path.isfile(args.checkpoint_dir + '/last.ckpt'):
        resume_from_checkpoint = args.checkpoint_dir + '/last.ckpt'
        print("LOADING 1:", resume_from_checkpoint)
    if args.resume_from_checkpoint != '':
        resume_from_checkpoint = args.resume_from_checkpoint
        print("LOADING 2:", resume_from_checkpoint)
    if args.resume_from_checkpoint_legacy != '':
        checkpoint = torch.load(args.resume_from_checkpoint_legacy)
        model.load_state_dict(checkpoint['state_dict'])
        print("LOADING 3:", args.resume_from_checkpoint_legacy)
    # Initialize trainer
    logger = TensorBoardLogger(args.checkpoint_dir, name='experiments', version=args.experiment_name)
    # model.set_logger(logger)
    early_stop_cb = EarlyStopping(patience=args.early_stop_threshold, verbose=True, monitor='val/mr')
    checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_top_k=3, mode='min', save_last=True,
                                          every_n_epochs=1,
                                          dirpath=args.checkpoint_dir, filename='checkpoint-{epoch:02d}')

    if args.viz or args.val:
        enable_checkpointing = False
        callbacks = []
    else:
        import json


import os
import torch
import pytorch_lightning as pl
import sys

from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from Forecasting.data.argoverse_datamodule import ArgoverseDataModule
from Forecasting.src.main import get_model, get_args
from pytorch_lightning.plugins import DDPPlugin


def parse_arguments():
    parser = ArgumentParser()

    # Load experiment and trainer-sepcific args
    parser = add_experimental_args(parser)
    # parser = pl.Trainer.add_argparse_args(parser)

    # Load dataset specific args
    parser = ArgoverseDataModule.add_data_specific_args(parser)

    # Load model specific args
    parser = get_args(parser.parse_known_args()[0].model_name, parser)

    args = parser.parse_args()

    # with open('args.json', 'r') as f:
    #     args.__dict__ = json.load(f)
    return args


def add_experimental_args(parent_parser):
    parser = ArgumentParser(parents=[parent_parser], add_help=False)

    # Parse dataset model to use
    parser.add_argument('--dataset', type=str, default='ArgoverseDatasetLaneGraph', help='Name of dataset to use')
    parser.add_argument('--model-name', type=str, required=True, help='Name of model to load')

    # General Params
    parser.add_argument("--mode", required=True, type=str,
                        # choices=['train', 'val', 'trainval', 'oracle-val', 'test', 'argoverse-test', 'sanity', 'sanity-test'],
                        help='Mode to run forecasting model in')
    parser.add_argument('--seed', type=int, help="Seed the random parameter generation")
    parser.add_argument('--no-batch-norm', dest='batch_norm', action='store_false',
                        help="Seed the random parameter generation")
    parser.add_argument("--viz", action='store_true', help='Visualize predictions')
    parser.add_argument("--test", action='store_true', help="test mode: doesn't shuffle train dataset")
    parser.add_argument("--val", action='store_true',
                        help="val mode: train without training doesn't shuffle train dataset")
    parser.add_argument("--plot-all-tracks", action='store_true', help='plot social actor tracks')
    parser.add_argument("--write_metrics_to_file", action='store_true',
                        help='write metrics of each scenario (ade / fde) to file')
    parser.add_argument("--comment", help='comment appended to end of visualizations filename')
    parser.add_argument("--path_to_interesting_scenario_list", default='', help='')

    # I/O and Feature Space Params
    parser.add_argument('--delta', action='store_true', help="Predict delta-xy coordinates instead of absolute")  # NOQA
    parser.add_argument('--training-signal', choices=['cond', 'focal', 'all'], default='cond',
                        help="focal: train only on focal agent's loss\ncond: train on social agents loss "
                             "(and condition on focal agent future GT)\nall: train on focal and social agents loss")
    parser.add_argument('--cond-all', action='store_true',
                        help="if true, will redraw training_signal every training step")
    parser.add_argument("--train-noise", action='store_true', help='Use rotational noise during training')
    parser.add_argument('--num-futures', type=int, default=6, help='Number of possible futures')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of gpus to use')

    # Datamodule Params
    parser.add_argument('--dataroot', required=True, help="Path to the processed dataset folder")
    parser.add_argument("--batch-size", type=int, default=25, help="Training batch size")
    parser.add_argument('--workers', type=int, default=8, help="Number of dataloader workers")

    # Trainer Params
    parser.add_argument("--gpus", type=int, default=1, help='# of GPUs to use for training')
    parser.add_argument("--check-val-every-n-epoch", type=int, default=1,
                        help="# of training epochs between val")  # NOQA
    parser.add_argument("--max-epochs", type=int, default=150, help="Max # of training epochs")
    parser.add_argument("--early-stop-threshold", type=int, default=20,
                        help="Number of consecutive val epochs without improvement before termination")  # NOQA
    parser.add_argument('--num-nodes', default=1, type=int, help='Number of nodes used')
    parser.add_argument('--precision', default=32, type=int, help='Precision employed in weights')
    parser.add_argument('--resume-from-checkpoint', default='', help='Path to checkpoint to resume training from')
    parser.add_argument('--distributed-backend', default='ddp', help='Trainer backend')
    parser.add_argument("--timesteps", type=int, default=50, help="Number of input timesteps")
    parser.add_argument("--outsteps", type=int, default=60, help="Number of output timesteps")

    # Logging Params
    parser.add_argument('--resume-from-checkpoint-legacy', default='',
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--checkpoint-dir', type=str, default=os.path.join(os.getcwd(), 'models'),
                        help='Path to save files')

    # Logging Params
    parser.add_argument('--experiment-name', type=str, help='Save file prefix')
    parser.add_argument('--logs-root', type=str, default=os.path.join(os.getcwd(), 'models') + '/experiments',
                        help='Path to save logs')

    return parser


def cli_main(args):
    # print(args)

    # Set global random seed
    pl.seed_everything(args.seed)

    # run with test run params if in ipython env; else with real run params
    try:
        __IPYTHON__
        in_ipdb = True
        print("TEST RUN: in ipdb")
    except NameError:  # no ipdb; real run
        in_ipdb = False
        print("not in ipdb")

    if not in_ipdb:
        plugins = DDPPlugin(find_unused_parameters=False)
        args.num_gpus = torch.cuda.device_count() if args.num_gpus is None else args.num_gpus
    else:
        plugins = []
        args.num_gpus = 0

    if args.test:
        sanity_val_steps = 0
        lim_train_batch = 1
        lim_val_batch = 1
        args.max_epochs = 10000
    else:
        lim_train_batch = None
        lim_val_batch = None
        sanity_val_steps = 1

    plugins = []
    args.num_gpus = 1

    # Initialize data module
    dm = ArgoverseDataModule(args)
    model = get_model(args.model_name, args)

    # Initialize trainer
    resume_from_checkpoint = None
    if os.path.isfile(args.checkpoint_dir + '/last.ckpt'):
        resume_from_checkpoint = args.checkpoint_dir + '/last.ckpt'
        print("LOADING 1:", resume_from_checkpoint)
    if args.resume_from_checkpoint != '':
        resume_from_checkpoint = args.resume_from_checkpoint
        print("LOADING 2:", resume_from_checkpoint)
    if args.resume_from_checkpoint_legacy != '':
        checkpoint = torch.load(args.resume_from_checkpoint_legacy)
        model.load_state_dict(checkpoint['state_dict'])
        print("LOADING 3:", args.resume_from_checkpoint_legacy)
    # Initialize trainer
    logger = TensorBoardLogger(args.checkpoint_dir, name='experiments', version=args.experiment_name)
    # model.set_logger(logger)
    early_stop_cb = EarlyStopping(patience=args.early_stop_threshold, verbose=True, monitor='val/mr')
    checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_top_k=3, mode='min', save_last=True,
                                          every_n_epochs=1,
                                          dirpath=args.checkpoint_dir, filename='checkpoint-{epoch:02d}')

    if args.viz or args.val:
        enable_checkpointing = False
        callbacks = []
    else:
        enable_checkpointing = True
        callbacks = [checkpoint_callback, early_stop_cb]

    check_val_every_n_epoch = args.check_val_every_n_epoch
    print("TRAINNG SIG", args.training_signal, '\n')
    trainer = pl.Trainer(check_val_every_n_epoch=check_val_every_n_epoch,  # log_every_n_steps=1,
                         gpus=args.num_gpus, plugins=plugins, num_sanity_val_steps=sanity_val_steps,
                         limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                         max_epochs=args.max_epochs, default_root_dir=args.logs_root, num_nodes=args.num_nodes,
                         precision=args.precision,  # resume_from_checkpoint=resume_from_checkpoint,
                         logger=logger, callbacks=callbacks, enable_checkpointing=enable_checkpointing,
                         gradient_clip_val=(args.gradient_clipping_val if args.gradient_clipping else 0.0), )

    if 'train' in args.mode:
        # if args.mode == 'train' or args.mode == 'trainval' or args.mode == 'sanity':
        trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
    elif 'test' in args.mode or 'val' in args.mode or 'cond' in args.mode:
        trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
    else:
        raise NotImplementedError

    print(args.training_signal)


if __name__ == '__main__':
    args = parse_arguments()
    if args.viz:
        args.batch_size = 1
    cli_main(args)

    enable_checkpointing = True
    callbacks = [checkpoint_callback, early_stop_cb]

check_val_every_n_epoch = args.check_val_every_n_epoch
print("TRAINNG SIG", args.training_signal, '\n')
trainer = pl.Trainer(check_val_every_n_epoch=check_val_every_n_epoch,  # log_every_n_steps=1,
                     gpus=args.num_gpus, plugins=plugins, num_sanity_val_steps=sanity_val_steps,
                     limit_val_batches=lim_val_batch, limit_train_batches=lim_train_batch,
                     max_epochs=args.max_epochs, default_root_dir=args.logs_root, num_nodes=args.num_nodes,
                     precision=args.precision,  # resume_from_checkpoint=resume_from_checkpoint,
                     logger=logger, callbacks=callbacks, enable_checkpointing=enable_checkpointing,
                     gradient_clip_val=(args.gradient_clipping_val if args.gradient_clipping else 0.0), )

if 'train' in args.mode:
    # if args.mode == 'train' or args.mode == 'trainval' or args.mode == 'sanity':
    trainer.fit(model, dm, ckpt_path=resume_from_checkpoint)
elif 'test' in args.mode or 'val' in args.mode or 'cond' in args.mode:
    trainer.test(model, datamodule=dm, ckpt_path=resume_from_checkpoint)
else:
    raise NotImplementedError

print(args.training_signal)

if __name__ == '__main__':
    args = parse_arguments()
    if args.viz:
        args.batch_size = 1
    cli_main(args)

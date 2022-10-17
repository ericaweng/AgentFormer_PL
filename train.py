import glob
import os
import sys
import argparse
import time
import subprocess
import numpy as np
import torch
import wandb
from torch import optim
from torch.utils.tensorboard import SummaryWriter
torch.autograd.set_detect_anomaly(True)

sys.path.append(os.getcwd())
from data.dataloader import data_generator
from model.model_lib import model_dict
from utils.torch import *
from utils.config import Config
from utils.utils import prepare_seed, print_log, AverageMeter, convert_secs2time, get_timestring

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def logging(cfg, log, epoch, total_epoch, iter, total_iter, ep, seq, frame, losses_str, sfm_hparams=None):
    eta = convert_secs2time(ep / iter * (total_iter * (total_epoch - epoch) - iter))
    print_log(f'{cfg} | Epo: {epoch:02d}/{total_epoch:02d} It: {iter:04d}/{total_iter:04d} '
              f'EP: {convert_secs2time(ep):s}, ETA: {eta:s}, seq {seq:s}, frame {frame:05d}, '
              f'{losses_str}, {sfm_hparams}', log)


def train(epoch):
    global tb_ind
    since_train = time.time()
    generator.shuffle()
    train_loss_meter = {x: AverageMeter() for x in cfg.loss_cfg.keys()}
    train_loss_meter['total_loss'] = AverageMeter()
    last_generator_index = 0
    seq, frame = None, None
    while not generator.is_epoch_end():
        data = generator()
        if data is not None:
            seq, frame = data['seq'], data['frame']
            model.set_data(data)
            model_data = model()
            total_loss, loss_dict, loss_unweighted_dict = model.compute_loss()
            """ optimize """
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            train_loss_meter['total_loss'].update(total_loss.item())
            for key in loss_unweighted_dict.keys():
                train_loss_meter[key].update(loss_unweighted_dict[key])

        # print("sfm_learnable_hparams:")
        # print(model.sfm_learnable_hparams)

        if generator.index - last_generator_index > cfg.print_freq:
            ep = time.time() - since_train
            losses_str = ' '.join([f'{x}: {y.avg:.3f} ({y.val:.3f})' for x, y in train_loss_meter.items()])
            if model.sfm_learnable_hparams is not None:
                learable_hparams_str = ' '.join([f'{k}: {v.item():.4f}' for k, v in model.sfm_learnable_hparams.items()])
            else:
                learable_hparams_str = None
            logging(cfg.id, log, epoch, cfg.num_epochs, generator.index, generator.num_total_samples, ep, seq, frame,
                    losses_str, learable_hparams_str)
            for name, meter in train_loss_meter.items():
                tb_logger.add_scalar('model_' + name, meter.avg, tb_ind)
            # if isinstance(model.sfm_learnable_hparams, dict):
            if model.sfm_learnable_hparams is not None:
                for name, param in model.sfm_learnable_hparams.items():
                    tb_logger.add_scalar('param_' + name, param, tb_ind)
            tb_ind += 1
            last_generator_index = generator.index

    scheduler.step()
    model.step_annealer()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='eth_agentformer_sfm_pre8-2')
    parser.add_argument('--cached', action='store_true', default=False)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--tmp', action='store_true', default=False)
    parser.add_argument('--eval_when_train', action='store_true', default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--weight', type=float, default=None)
    parser.add_argument('--sigma_d', type=float, default=None)

    args = parser.parse_args()

    """ setup """
    cfg = Config(args.cfg, args.tmp, create_dirs=True, additional_cfg_vars={'weight': args.weight, 'sigma_d': args.sigma_d})
    prepare_seed(cfg.seed)
    torch.set_default_dtype(torch.float32)
    device = torch.device('cuda', index=args.gpu) if torch.cuda.is_available() else torch.device('cpu')
    torch.cuda.set_device(args.gpu)
    if torch.cuda.is_available(): torch.cuda.set_device(args.gpu)

    time_str = get_timestring()
    log = open(os.path.join(cfg.log_dir, 'log.txt'), 'a+')
    print_log("time str: {}".format(time_str), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch version : {}".format(torch.__version__), log)
    print_log("cudnn version : {}".format(torch.backends.cudnn.version()), log)
    tb_logger = SummaryWriter(cfg.tb_dir)
    tb_ind = 0

    """ data """
    generator = data_generator(cfg, log, split='train', phase='training')

    """ model """
    model_id = cfg.get('model_id', 'agentformer')
    model = model_dict[model_id](cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler_type = cfg.get('lr_scheduler', 'linear')
    if scheduler_type == 'linear':
        scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.lr_fix_epochs, nepoch=cfg.num_epochs)
    elif scheduler_type == 'step':
        scheduler = get_scheduler(optimizer, policy='step', decay_step=cfg.decay_step, decay_gamma=cfg.decay_gamma)
    else:
        raise ValueError('unknown scheduler type!')

    """resume training"""
    if args.cached or args.start_epoch > 0:
        if args.start_epoch > 0:
            cp_path = cfg.model_path % args.start_epoch
        elif args.cached:
            existing_models = glob.glob('/'.join(cfg.model_path.split('/')[:-1]) + '/*')
            if os.path.exists(cfg.model_path_last):
                cp_path = cfg.model_path_last
            elif len(existing_models):
                cp_path = existing_models[-1]
            else:
                raise NotImplementedError
            # cp_path = cfg.model_path_last
            print("loading existing model from:", cp_path)
        else:
            raise NotImplementedError
        print_log(f'loading model from checkpoint: {cp_path}', log)
        model_cp = torch.load(cp_path, map_location='cpu')
        if args.cached:
            args.start_epoch = model_cp['epoch']
        model.load_state_dict(model_cp['model_dict'])
        if 'opt_dict' in model_cp:
            optimizer.load_state_dict(model_cp['opt_dict'])
            # optimizer = optimizer.to(device)
            optimizer_to(optimizer, device)
        if 'scheduler_dict' in model_cp:
            scheduler.load_state_dict(model_cp['scheduler_dict'])
            # optimizer_to(scheduler, device)
            # scheduler = scheduler.to(device)
        model.set_device(device)
        print("device:", device)

    """ start training """
    print("Start Training")
    model.set_device(device)
    model.train()
    for i in range(args.start_epoch, cfg.num_epochs):
        print("training for:", cfg.num_epochs, "epochs")
        train(i)
        """ save last model """
        model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(),
                    'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
        torch.save(model_cp, cfg.model_path_last)

        """ save model """
        if cfg.model_save_freq > 0 and (i + 1) % cfg.model_save_freq == 0:
            cp_path = cfg.model_path % (i + 1)
            # model_cp = {'model_dict': model.state_dict(), 'opt_dict': optimizer.state_dict(), 'scheduler_dict': scheduler.state_dict(), 'epoch': i + 1}
            torch.save(model_cp, cp_path)

            if args.eval_when_train:
                # cmd = f"python test.py --cfg {args.cfg} --gpu {args.gpu} --data_eval test --epochs {i + 1}"
                cmd = f"python test.py --cfg {args.cfg} --gpu {args.gpu} --data_eval test --epochs {i + 1} --weight {args.weight} --sigma_d {args.sigma_d}"
                subprocess.Popen(cmd.split(' '))

    """ testing """
    if not args.eval_when_train:
        del model
        torch.cuda.empty_cache()
        test_epochs = ','.join([str(x) for x in range(cfg.model_save_freq, cfg.num_epochs + 1, cfg.model_save_freq)])
        cmd = f"python test.py --cfg {args.cfg} --gpu {args.gpu} --data_eval test --epochs {test_epochs} --weight {args.weight} --sigma_d {args.sigma_d}"
        subprocess.run(cmd.split(' '))

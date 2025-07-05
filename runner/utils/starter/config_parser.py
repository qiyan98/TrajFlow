# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import datetime
import os
from pathlib import Path, PosixPath
import copy
import git
import shutil
import yaml
from easydict import EasyDict

import wandb

import torch

from trajflow.config import init_cfg, cfg_from_yaml_file, log_config_to_file
from trajflow.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    """basic configs"""
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--logger_iter_interval', type=int, default=10, help='logger info interval')

    """optimizaion parameters"""
    parser.add_argument('--batch_size', type=int, default=None, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='number of epochs to train for')
    parser.add_argument('--learning_rate', default=None, type=float, help='Overwrite the learning rate.')
    parser.add_argument('--lr_scheduler', default=None, type=str, choices=['cosine', 'lambdaLR', 'linearLR', 'constant'], help='Overwrite the LR scheduler.')
    parser.add_argument('--weight_decay', default=None, type=float, help='Overwrite the weight decay.')
    parser.add_argument('--ema_coef', default=None, type=float, help='Overwrite the EMA coefficient.')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    """random seed control"""
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='fix random seed for reproducibility')

    """checkpoint loading, saving and evaluation"""
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--ckpt_save_interval', type=int, default=2, help='save checkpoint every few number of training epochs')
    parser.add_argument('--ckpt_save_time_interval', type=int, default=600, help='save checkpoint every few seconds')
    parser.add_argument('--max_ckpt_save_num', type=int, default=5, help='max number of saved checkpoint')
    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes for ckpt evaluation')

    """DDP configs"""
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')

    args = parser.parse_args()

    """load config"""
    cfg = init_cfg()
    cfg_from_yaml_file(args.cfg_file, cfg)

    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.launcher == 'none':
        cfg.OPT.DIST_TRAIN, cfg.OPT.TOTAL_GPUS, cfg.OPT.WITHOUT_SYNC_BN = False, 1, True
    elif args.launcher == 'pytorch':
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        cfg.OPT.TOTAL_GPUS, cfg.LOCAL_RANK = common_utils.init_dist_pytorch(local_rank, backend='nccl')
        cfg.OPT.DIST_TRAIN, cfg.OPT.WITHOUT_SYNC_BN = True, False
    else:
        raise ValueError('Invalid launcher: %s' % args.launcher)
    
    if args.batch_size is not None:
        assert args.batch_size % cfg.OPT.TOTAL_GPUS == 0, 'Batch size should match the number of gpus'
        cfg.OPT.BATCH_SIZE_PER_GPU = args.batch_size // cfg.OPT.TOTAL_GPUS
    for param, attr in [('epochs', 'NUM_EPOCHS'), ('learning_rate', 'LR'), ('lr_scheduler', 'SCHEDULER'), 
                       ('weight_decay', 'WEIGHT_DECAY'), ('ema_coef', 'EMA_COEF')]:
        if getattr(args, param) is not None:
            setattr(cfg.OPT, attr, getattr(args, param))
    cfg.OPT.WORKERS = args.workers
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.DEVICE = cfg.MODEL_DMT.DEVICE = device

    return args, cfg


def init_basics():
    ###### Start of Init ######

    """Parse arguments and config"""
    args, cfg = parse_config()
    
    """Set random seed"""
    if args.fix_random_seed:
        common_utils.set_random_seed(42)

    """Set up saving folder"""
    # note important configs in the folder name
    tag_parts = []
    if cfg.MODEL_DMT.DMT.DROPOUT:
        tag_parts.append(f'_DO{cfg.MODEL_DMT.DMT.DROPOUT:.2f}')
    if cfg.OPT.WEIGHT_DECAY:
        tag_parts.append(f'_WD{cfg.OPT.WEIGHT_DECAY:.0e}')
    tag_parts.append(f'_BS{cfg.OPT.BATCH_SIZE_PER_GPU * cfg.OPT.TOTAL_GPUS}_EP{cfg.OPT.NUM_EPOCHS}')
    default_tag = ''.join(tag_parts).replace('__', '_')
    args.extra_tag = '_'.join([args.extra_tag, default_tag]).replace('__', '_')

    """Initialize place holder saving folders and logger"""
    output_dir = os.path.join(cfg.ROOT_DIR, 'output', cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    eval_output_dir = os.path.join(output_dir, 'eval', 'eval_with_train')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)

    cfg.SAVE_DIR = EasyDict({
        'OUTPUT_DIR': output_dir,
        'CKPT_DIR': ckpt_dir,
        'EVAL_OUTPUT_DIR': eval_output_dir
    })

    log_file = os.path.join(output_dir, 'log_train_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if cfg.OPT.DIST_TRAIN:
        logger.info('total_batch_size: %d' % (cfg.OPT.TOTAL_GPUS * cfg.OPT.BATCH_SIZE_PER_GPU))

    logger.info('**********************Argparser**********************')
    for key, val in vars(args).items():
        logger.info('{:32} {}'.format(key, val))

    logger.info('**********************Configurations**********************')
    log_config_to_file(copy.deepcopy(cfg), logger=logger)

    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))  # copy original config
        # dump the updated config from easydict [not perfect as there is special items in the original config like &object_type]

        def easydict_to_dict(easydict_obj):
            # Function to convert EasyDict to a dictionary recursively
            result = {}
            for key, value in easydict_obj.items():
                if isinstance(value, EasyDict):
                    result[key] = easydict_to_dict(value)
                else:
                    if isinstance(value, PosixPath):
                        result[key] = os.path.abspath(value)  # convert PosixPath to string
                    else:
                        result[key] = value
            return result
        
        nested_dict = easydict_to_dict(cfg)
        with open(os.path.join(output_dir, '{:s}_updated.yaml'.format(os.path.basename(args.cfg_file)[:-5])), 'w') as f:
            yaml.dump(nested_dict, f)

    # wandb log
    wb_log = None
    if cfg.LOCAL_RANK == 0:
        # Initialize wandb run for training
        wb_log = wandb.init(
            project="trajflow",
            name=f"{cfg.TAG}_{args.extra_tag}",
            config=cfg,
            dir=output_dir
        )

    # save version control information
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    logger.info("git hash: {}".format(sha))

    # backup code
    code_backup_dir = os.path.join(output_dir, 'code_backup')
    shutil.rmtree(code_backup_dir, ignore_errors=True)
    os.makedirs(code_backup_dir, exist_ok=True)
    if cfg.LOCAL_RANK == 0:
        dirs_to_save = ['trajflow', 'runner']
        for this_dir in dirs_to_save:
            src_dir = os.path.join(cfg.ROOT_DIR, this_dir)
            dest_dir = os.path.join(code_backup_dir, this_dir)
            
            if os.path.exists(src_dir):
                try:
                    shutil.copytree(src_dir, dest_dir)
                    logger.info(f"Successfully copied {src_dir} to {dest_dir}")
                except (shutil.Error, OSError) as e:
                    logger.error(f"Error copying {src_dir} to {dest_dir}: {e}")
            else:
                logger.warning(f"Source directory {src_dir} does not exist. Skipping.")

        # [shutil.copytree(os.path.join(cfg_diff.ROOT_DIR, this_dir), os.path.join(code_backup_dir, this_dir)) for this_dir in dirs_to_save]
        logger.info("Code is backedup to {}".format(code_backup_dir))

    ###### End of Init ######

    return args, cfg, logger, wb_log

# Copyright (c) 2025-present, Qi Yan.
# Copyright (c) Shaoshuai Shi.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the Motion Transformer (https://arxiv.org/abs/2209.13508) implementation
# from https://github.com/sshaoshuai/MTR by Shaoshuai Shi, Li Jiang, Dengxin Dai, Bernt Schiele
####################################################################################


import _init_path
import argparse
import datetime
from glob import glob
import os
import re
from pathlib import Path
import copy
from easydict import EasyDict


import torch

from trajflow.datasets import build_dataloader
from trajflow.config import init_cfg, cfg_from_yaml_file, log_config_to_file
from trajflow.utils import common_utils
from utils.starter.network import init_network
from utils.eval import eval_single_ckpt, repeat_eval_ckpt


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    """basic configs"""
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for testing')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--logger_iter_interval', type=int, default=10, help='logger info interval')

    """optimizaion parameters"""
    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for testing')
    parser.add_argument('--ema_coef', default='1.0', nargs='+', help='To use EMA version weight with specified coefficients.')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')

    """random seed control"""
    parser.add_argument('--seed', type=int, default=None, help='random seed')

    """checkpoint loading, saving and evaluation"""
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')

    """DDP configs"""
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='none')

    """ablations"""
    # FM sampler
    parser.add_argument('--fm_sampling_steps', default=None, type=int, help='number of diffusion steps for FM')

    # general evaluation settings
    parser.add_argument('--full_eval', action='store_true', default=False, help='use full validation set for evaluation')

    # Evaluation dataset selection
    parser.add_argument('--val', action='store_true', default=False, help='get results on validation set')
    parser.add_argument('--test', action='store_true', default=False, help='get results on test set')

    parser.add_argument('--interactive', action='store_true', default=False, help='get results on the interactive split')

    # submission meta-data
    parser.add_argument('--submit', action='store_true', default=False, help='submit the results')
    parser.add_argument('--email', type=str, default=None, help='email for submission')
    parser.add_argument('--method_nm', type=str, default='anonymous', help='unique method name')

    args = parser.parse_args()

    assert sum([args.val, args.test]) == 1, 'Choose exactly one dataset to evaluate on (val or test)'

    if args.submit:
        assert args.email is not None, 'Please provide an email for submission'

        ckpt_basename = os.path.basename(args.ckpt)
        if ckpt_basename.startswith('checkpoint_epoch'):
            ckpt_info = 'ep_{:s}'.format(ckpt_basename.split('_')[2].replace('.pth', ''))
        elif ckpt_basename.endswith('_model.pth'):
            ckpt_info = 'ckpt_{:s}'.format(ckpt_basename.split('_')[0])
        else:
            raise ValueError('Invalid checkpoint name: %s' % ckpt_basename)
        
        args.method_nm += '_{:s}'.format(ckpt_info)

    """arg parsing special conditions"""
    # auto detect cfg_file if not specified
    if args.cfg_file is None:
        if os.path.exists(args.ckpt) and args.ckpt.endswith('.pth'):
            _yaml_files = glob(os.path.join(os.path.abspath(os.path.join(args.ckpt, '../..')), '*.yaml'))
            args.cfg_file = [yaml for yaml in _yaml_files if yaml.endswith('_updated.yaml')][0]
            assert os.path.exists(args.cfg_file), 'Config file not found: {}'.format(args.cfg_file)
            print("Loading auto-detected config file: {}".format(args.cfg_file))

    # handle special ema_coef keywords 'all' or 'none'
    _ema_coef = args.ema_coef
    if (isinstance(_ema_coef, list) and len(_ema_coef) == 1) or isinstance(_ema_coef, str):
        # either 'all', 'none' or a single value; it must be a string
        _ema_coef = _ema_coef[0] if isinstance(_ema_coef, list) else _ema_coef
        assert isinstance(_ema_coef, str)
        if _ema_coef in ['all', 'none']:
            args.ema_coef = None if _ema_coef == 'none' else 'all'
        else:
            args.ema_coef = [float(_ema_coef)]
    else:
        # specific EMA coefficients
        _ema_coef = []
        for item in args.ema_coef:
            # store float number except for special keywords 'all' or 'none'
            _ema_coef.append(float(item) if item not in ['all', 'none'] else item)
        args.ema_coef = _ema_coef  # always a list

    """build config"""
    cfg = init_cfg()                                    # init
    cfg_empty = copy.deepcopy(cfg)                      # backup
    cfg_from_yaml_file(args.cfg_file, cfg)              # load cfg from file 
    cfg.TAG = Path(args.cfg_file).stem                  # reset TAG
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.replace(str(cfg_empty['ROOT_DIR']), '').split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.ROOT_DIR = cfg_empty['ROOT_DIR']                # restore ROOT_DIR
    del cfg.SAVE_DIR                                    # remove SAVE_DIR

    # DDP settings
    if args.launcher == 'none':
        cfg.OPT.DIST_TRAIN = False
        cfg.OPT.TOTAL_GPUS = 1
        cfg.OPT.WITHOUT_SYNC_BN = True
    elif args.launcher == 'pytorch':
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        cfg.OPT.TOTAL_GPUS, cfg.LOCAL_RANK = common_utils.init_dist_pytorch(local_rank, backend='nccl')
        cfg.OPT.DIST_TRAIN = True
        cfg.OPT.WITHOUT_SYNC_BN = False
    else:
        raise ValueError('Invalid launcher: %s' % args.launcher)
    
    ### Optimization hyperparameters ###
    if args.batch_size is not None:
        assert args.batch_size % cfg.OPT.TOTAL_GPUS == 0, 'Batch size should match the number of gpus'
        cfg.OPT.BATCH_SIZE_PER_GPU = args.batch_size // cfg.OPT.TOTAL_GPUS
    if args.ema_coef is not None:
        cfg.OPT.EMA_COEF = args.ema_coef
    cfg.OPT.WORKERS = args.workers
    ### Optimization hyperparameters ###

    # overwrite configs
    if args.fm_sampling_steps is not None:
        cfg.MODEL_DMT.DENOISING.FM.SAMPLING_STEPS = args.fm_sampling_steps

    if args.full_eval:
        cfg.DATA_CONFIG.SAMPLE_INTERVAL.eval = 1
        cfg.DATA_CONFIG.SAMPLE_INTERVAL.test = 1
        cfg.DATA_CONFIG.SAMPLE_INTERVAL.inter_eval = 1
        cfg.DATA_CONFIG.SAMPLE_INTERVAL.inter_test = 1
        args.extra_tag += '_full_eval'
        
    """get output dir"""
    output_dir_prefix = ''
    if args.val:
        output_dir_prefix += 'val_'
    elif args.test:
        output_dir_prefix += 'test_'
    else:
        raise NotImplementedError
    if args.interactive:
        output_dir_prefix += 'inter_'
    if args.submit:
        output_dir_prefix += args.email.replace('@', '_').replace('.', '_') + '_' + args.method_nm

    output_dir_nm = output_dir_prefix + '_' + args.extra_tag
    output_dir_nm = output_dir_nm.replace('__', '_')
    cfg.OUTPUT_DIR_PREFIX = output_dir_prefix

    if 'output/' in cfg.EXP_GROUP_PATH:
        output_dir = cfg.ROOT_DIR / cfg.EXP_GROUP_PATH / output_dir_nm
    else:
        output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / output_dir_nm
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_output_dir = os.path.join(output_dir,  'eval')
    if not args.eval_all:
        if 'latest_model' in args.ckpt:
            epoch_id = None
            eval_output_dir = os.path.join(eval_output_dir, 'latest_model')
        else:
            num_list = re.findall(r'\d+', args.ckpt) if args.ckpt is not None else []
            epoch_id = num_list[-2] if num_list.__len__() > 0 else 'no_number'
            eval_output_dir = os.path.join(eval_output_dir, 'epoch_%s' % epoch_id)
    else:
        epoch_id = None
        eval_output_dir = os.path.join(eval_output_dir, 'eval_all_default')

    os.makedirs(eval_output_dir, exist_ok=True)

    cfg.SAVE_DIR = EasyDict({'OUTPUT_DIR': output_dir,
                             'EVAL_OUTPUT_DIR': eval_output_dir,
                             'CKPT_DIR': os.path.join(output_dir, 'ckpt')})
    
    """init logger"""
    log_file = os.path.join(eval_output_dir, 'log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    """set random seed"""
    if args.seed:
        logger.info('Set random seed to %d' % args.seed)
        common_utils.set_random_seed(args.seed)

    return args, cfg, logger


def main():
    
    """Init"""
    args, cfg, logger = parse_config()

    if args.submit:
        method_name = args.method_nm
        if len(method_name) > 25:
            method_name = method_name[:25]
        submission_info = dict(
            account_name=args.email,
            unique_method_name=method_name,
            authors=['anonymous'],
            affiliation='anonymous',
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            public_model_names='N/A',
            num_model_parameters='N/A',
        )
    else:
        submission_info = None

    """build dataloader"""
    test_batch_size = cfg.OPT.TOTAL_GPUS * cfg.OPT.BATCH_SIZE_PER_GPU

    if cfg.OPT.DIST_TRAIN:
        logger.info('total_batch_size: %d' % (cfg.OPT.TOTAL_GPUS * cfg.OPT.BATCH_SIZE_PER_GPU))

    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=test_batch_size,
        dist=cfg.OPT.DIST_TRAIN, workers=cfg.OPT.WORKERS, 
        logger=logger, training=False,
        testing=args.test, inter_pred=args.interactive)


    """build model"""
    model, denoiser = init_network(cfg, logger)

    """log to file"""
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))

    log_config_to_file(cfg, logger=logger)

    """start evaluation"""
    with torch.no_grad():
        if args.eval_all:
            repeat_eval_ckpt(denoiser, test_loader, cfg, args, logger, args_ema_coef=args.ema_coef, submission_info=submission_info)
        else:
            eval_single_ckpt(denoiser, test_loader, cfg, args, logger, args_ema_coef=args.ema_coef, submission_info=submission_info)


if __name__ == '__main__':
    main()

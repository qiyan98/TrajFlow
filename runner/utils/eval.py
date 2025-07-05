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
import glob
import os
import re
import time
import copy
import torch
import wandb

from .tester import eval_one_epoch


def get_ema_weight_keywords(ckpt_keys, ema_coef, logger):
    """Get EMA weight keywords based on checkpoint data and EMA coefficients."""
    model_keys = [key for key in ckpt_keys if key.startswith('model_')]
    online_key = 'model_state'
    weight_keywords = [online_key]
    
    if ema_coef is None:
        logger.info('Not using EMA weight.')
    elif ema_coef == 'all':
        weight_keywords = model_keys
        logger.info('Use all possible EMA weights.')
    else:
        logger.info(f'Using EMA weight with coefficients: {ema_coef}')
        if 1.0 not in ema_coef:
            weight_keywords.remove(online_key)
        else:
            ema_coef.remove(1.0)
        
        for coef in ema_coef:
            weight_key = f'model_ema_beta_{coef:.4f}'
            assert weight_key in model_keys, f"{weight_key} not found in model data!"
            weight_keywords.append(weight_key)
    
    logger.info(f'Model weights to load: {weight_keywords}')
    return weight_keywords


def eval_single_ckpt(denoiser, test_loader, cfg, args, logger, args_ema_coef=None, submission_info=None):
    """Evaluate a single checkpoint with optional EMA variants."""
    cfg_ = copy.deepcopy(cfg)
    dist_test = cfg.OPT.DIST_TRAIN
    eval_output_dir = cfg.SAVE_DIR.EVAL_OUTPUT_DIR
    
    # Load checkpoint into memory for EMA variants
    device = 'CPU' if dist_test else 'GPU'
    logger.info(f'==> Loading parameters from checkpoint {args.ckpt} to {device}')
    ckpt_state = torch.load(args.ckpt, map_location=torch.device('cpu') if dist_test else None)
    
    weight_keywords = get_ema_weight_keywords(ckpt_state.keys(), args_ema_coef, logger)
    
    for weight_kw in weight_keywords:
        # Load checkpoint
        if args.ckpt is not None:
            it, epoch = denoiser.model.load_params(
                ckpt_path=args.ckpt, to_cpu=dist_test, 
                ckpt_state=ckpt_state, optimizer=None, ema_model_kw=weight_kw
            )
            epoch += 1  # because the epoch is 0-indexed in the checkpoint
        else:
            it, epoch, ckpt_state = -1, -1, None
        
        denoiser.cuda()
        logger.info(f'*************** Successfully load model (epoch={epoch}, iter={it}, EMA weight_keyword={weight_kw}) for EVALUATION *****************')
        
        # Setup result directory
        base_dir = os.path.basename(eval_output_dir)
        if base_dir == 'default':
            result_dir = os.path.join(base_dir, f'weight_{weight_kw}_epoch_{epoch}')
        else:
            result_dir = os.path.join(eval_output_dir, f'weight_{weight_kw}_epoch_{epoch}')
        
        os.makedirs(result_dir, exist_ok=True)
        logger.info(f'*************** Saving results to {result_dir} *****************')
        
        # Run evaluation
        cfg = copy.deepcopy(cfg_)
        cfg.SAVE_DIR.EVAL_OUTPUT_DIR = result_dir
        
        eval_one_epoch(
            denoiser, test_loader, cfg, epoch, logger,
            inter_pred=args.interactive, flag_submission=args.submit, 
            submission_info=submission_info, logger_iter_interval=args.logger_iter_interval
        )


def get_unevaluated_ckpt(ckpt_dir, record_file, start_epoch):
    """Find the next unevaluated checkpoint."""
    ckpt_files = glob.glob(os.path.join(ckpt_dir, '*checkpoint_epoch_*.pth'))
    ckpt_files.sort(key=os.path.getmtime)
    
    with open(record_file, 'r') as f:
        evaluated_epochs = {float(x.strip()) for x in f.readlines()}
    
    for ckpt_path in ckpt_files:
        match = re.search(r'checkpoint_epoch_(.*?)\.pth', ckpt_path)
        if not match or 'optim' in match.group(1):
            continue
        
        try:
            epoch_id = float(match.group(1))
        except:
            epoch_id = float(match.group(1).split('_')[0])
        
        if epoch_id not in evaluated_epochs and int(epoch_id) >= start_epoch:
            return int(epoch_id), ckpt_path
    
    return -1, None


def repeat_eval_ckpt(denoiser, test_loader, cfg, args, logger, args_ema_coef=None, submission_info=None):
    """Repeatedly evaluate checkpoints as they become available."""
    if args_ema_coef is not None:
        raise NotImplementedError('EMA checkpoint variants not supported for repeated evaluation')
    
    cfg_ = copy.deepcopy(cfg)
    dist_test = cfg.OPT.DIST_TRAIN
    eval_output_dir = cfg.SAVE_DIR.EVAL_OUTPUT_DIR
    ckpt_dir = cfg.SAVE_DIR.CKPT_DIR
    
    # Setup checkpoint record file
    record_file = os.path.join(eval_output_dir, 'eval_list_val.txt')
    open(record_file, 'a').close()  # Create file if doesn't exist
    
    # Setup wandb logging - use the same wandb run as training
    wb_log = None
    if cfg.LOCAL_RANK == 0:
        # Get the current wandb run instead of creating a new one
        wb_log = wandb.run
    
    total_time = 0
    wait_seconds = 10
    
    while True:
        # Find next unevaluated checkpoint
        epoch_id, ckpt_path = get_unevaluated_ckpt(ckpt_dir, record_file, args.start_epoch)
        
        if epoch_id == -1:
            # No checkpoint found, wait and retry
            if cfg.LOCAL_RANK == 0:
                progress = total_time / 60
                print(f'Wait {wait_seconds}s for next check (progress: {progress:.1f}/{args.max_waiting_mins} min): {ckpt_dir}\r', 
                      end='', flush=True)
            
            time.sleep(wait_seconds)
            total_time += wait_seconds
            
            if total_time >= args.max_waiting_mins * 60:
                break
            continue
        
        # Reset timer and evaluate checkpoint
        total_time = 0
        
        it, epoch = denoiser.model.load_params(ckpt_path=ckpt_path, to_cpu=dist_test)
        logger.info(f'*************** LOAD MODEL (epoch={epoch}, iter={it}) for EVALUATION *****************')
        denoiser.cuda()
        
        # Setup result directory and run evaluation
        result_dir = os.path.join(eval_output_dir, f'epoch_{epoch_id:02d}')
        os.makedirs(result_dir, exist_ok=True)
        
        cfg = copy.deepcopy(cfg_)
        cfg.SAVE_DIR.EVAL_OUTPUT_DIR = result_dir
        
        wb_dict = eval_one_epoch(
            denoiser, test_loader, cfg, epoch_id, logger,
            inter_pred=args.interactive, flag_submission=args.submit,
            submission_info=submission_info, logger_iter_interval=args.logger_iter_interval
        )
        
        # Log to wandb
        if wb_log:
            wb_dict = {k: v for k, v in wb_dict.items() if '-----' not in k}  # skip meaningless entries
            eval_log_dict = {f'eval/{key}': val for key, val in wb_dict.items()}
            eval_log_dict['epoch'] = epoch_id
            wb_log.log(eval_log_dict)
        
        # Record evaluated epoch
        with open(record_file, 'a') as f:
            f.write(f'{epoch_id}\n')
        logger.info(f'Epoch {epoch_id} has been evaluated')


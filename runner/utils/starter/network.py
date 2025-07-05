# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import copy
import torch
import torch.optim.lr_scheduler as lr_sched
from glob import glob
from ema_pytorch import EMA
from trajflow.models.dmt_model import DenoisingMotionTransformer
from trajflow.denoising.flow_matching import FlowMatcher


def init_network(cfg, logger):
    """
    Initialize the networks.
    """

    # build model
    model = DenoisingMotionTransformer(config=cfg, logger=logger)
    if cfg.OPT.DIST_TRAIN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(cfg.DEVICE)

    # Build diffusion objective
    denoising_cfg = cfg.MODEL_DMT.DENOISING
    
    denoiser = FlowMatcher(
        model=model,
        sampling_timesteps=denoising_cfg.FM.SAMPLING_STEPS,
        objective=denoising_cfg.FM.OBJECTIVE,
        t_schedule=denoising_cfg.FM.T_SCHEDULE,
        logger=logger,
        data_rescale=cfg.DATA_CONFIG.DATA_RESCALE,
        ckpt_dir=cfg.SAVE_DIR.CKPT_DIR,
        model_cfg=copy.deepcopy(cfg.MODEL_DMT)
    )
    denoiser.to(cfg.DEVICE)
    
    return model, denoiser


def init_optimizer(model, opt_cfg):
    """Initialize optimizer."""
    optimizers = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW
    }
    
    if opt_cfg.OPTIMIZER not in optimizers:
        raise NotImplementedError(f"Optimizer {opt_cfg.OPTIMIZER} not implemented.")
    
    return optimizers[opt_cfg.OPTIMIZER](
        model.parameters(), 
        lr=opt_cfg.LR, 
        weight_decay=opt_cfg.get('WEIGHT_DECAY', 0)
    )


def init_ema_helper(model, opt_cfg, logger):
    """Setup exponential moving average training helper."""
    ema_coef = opt_cfg.EMA_COEF
    
    # Determine if EMA should be used
    if isinstance(ema_coef, float):
        flag_ema = ema_coef < 1
        ema_coef = [ema_coef] if flag_ema else None
    elif isinstance(ema_coef, list):
        flag_ema = True
    else:
        flag_ema = False
        ema_coef = None
    
    if not flag_ema:
        logger.info("Exponential moving average is OFF.")
        return None
    
    # Create EMA helpers
    ema_helper = [
        EMA(model=model, beta=coef, update_every=1, update_after_step=0, inv_gamma=1, power=1)
        for coef in sorted(ema_coef)
    ]
    logger.info(f"Exponential moving average is ON. Coefficient: {ema_coef}")
    return ema_helper


def load_checkpoint(model, optimizer, ema_helper, logger, ckpt_path, ckpt_dir):
    """
    Load checkpoint if it is possible.
    """
    start_epoch = it = 0
    last_epoch = -1

    if ckpt_path is not None:
        # Load checkpoint from specified path
        it, start_epoch = model.load_params(ckpt_path, optimizer=optimizer, ema_helper=ema_helper)
        last_epoch = start_epoch + 1
        return it, start_epoch, last_epoch
    
    # Load latest checkpoint from directory
    ckpt_list = glob(os.path.join(ckpt_dir, '*.pth'))
    if not ckpt_list:
        logger.info("No checkpoint found. Training from scratch.")
        return it, start_epoch, last_epoch
    
    # Find and load the latest valid checkpoint
    ckpt_list.sort(key=os.path.getmtime)
    for ckpt_file in reversed(ckpt_list):
        if os.path.basename(ckpt_file) == 'best_model.pth':
            continue
        try:
            ckpt_state = torch.load(ckpt_file, map_location=torch.device('cpu'))
            it, start_epoch = model.load_params(ckpt_file, ckpt_state=ckpt_state, optimizer=optimizer, ema_helper=ema_helper)
            last_epoch = start_epoch + 1
            break
        except:
            continue
    
    return it, start_epoch, last_epoch


def init_scheduler(optimizer, opt_cfg, total_epochs, total_iters_each_epoch, last_epoch):
    """Initialize learning rate scheduler."""
    scheduler_type = opt_cfg.get('SCHEDULER', None)
    total_iterations = total_epochs * total_iters_each_epoch
    
    if scheduler_type == 'cosine':
        # Cosine annealing with linear warmup
        warmup_iterations = max(1, int(total_iterations * 0.05))
        warmup_scheduler = lr_sched.LambdaLR(
            optimizer, 
            lambda step: max(opt_cfg.LR_CLIP / opt_cfg.LR, step / warmup_iterations)
        )
        cosine_scheduler = lr_sched.CosineAnnealingLR(
            optimizer, 
            T_max=total_iterations - warmup_iterations, 
            eta_min=opt_cfg.LR_CLIP
        )
        scheduler = lr_sched.SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, cosine_scheduler], 
            milestones=[warmup_iterations]
        )
        
    elif scheduler_type == 'lambdaLR':
        # LambdaLR with decay steps
        decay_step_list = opt_cfg.get('DECAY_STEP_LIST', [22, 24, 26, 28])
        if len(decay_step_list) == 1 and decay_step_list[0] == -1:
            decay_step_list = [22, 24, 26, 28]
        
        decay_steps = [x * total_iters_each_epoch for x in decay_step_list]
        
        def lr_lambda(cur_step):
            cur_decay = 1
            for decay_step in decay_steps:
                if cur_step >= decay_step:
                    cur_decay *= opt_cfg.LR_DECAY
            return max(cur_decay, opt_cfg.LR_CLIP / opt_cfg.LR)
        
        scheduler = lr_sched.LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == 'linearLR':
        # LinearLR
        scheduler = lr_sched.LinearLR(
            optimizer, 
            start_factor=1.0, 
            end_factor=opt_cfg.LR_CLIP / opt_cfg.LR, 
            total_iters=total_iterations, 
            last_epoch=last_epoch
        )
        
    elif scheduler_type == 'constant':
        # Constant learning rate
        scheduler = lr_sched.LambdaLR(optimizer, lambda x: 1.0, last_epoch=last_epoch)
        
    else:
        raise NotImplementedError(f"Unsupported scheduler: {scheduler_type}")
    
    # Handle last_epoch for schedulers that don't support it properly
    if last_epoch > 0 and scheduler_type in ['cosine', 'lambdaLR']:
        for _ in range(last_epoch * total_iters_each_epoch):
            scheduler.step()
    
    return scheduler

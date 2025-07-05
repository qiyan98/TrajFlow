# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.nn.parallel import DistributedDataParallel

from runner.utils.starter.config_parser import init_basics
from runner.utils.starter.data import init_dataloader
from runner.utils.starter.network import init_network, init_optimizer, init_ema_helper, load_checkpoint, init_scheduler
from runner.utils.eval import repeat_eval_ckpt
from runner.utils.trainer import train_model


def main():
    """
    Main function.
    """

    """Init"""
    args, cfg, logger, wb_log = init_basics()


    """build dataloader"""
    _, train_loader, train_sampler, _, test_loader, _ = init_dataloader(cfg, logger)


    """build model"""
    model, denoiser = init_network(cfg, logger)


    """build optimizer"""
    optimizer = init_optimizer(model, cfg.OPT)
    ema_helper = init_ema_helper(model, cfg.OPT, logger)


    """load model checkpoint"""
    it, start_epoch, last_epoch = load_checkpoint(model, optimizer, ema_helper, logger, 
                                                  ckpt_path=args.ckpt, ckpt_dir=cfg.SAVE_DIR.CKPT_DIR)
    

    """build scheduler"""
    # build after loading ckpt since the optimizer may be changed
    scheduler = init_scheduler(optimizer, cfg.OPT, total_epochs=cfg.OPT.NUM_EPOCHS - start_epoch,
                               total_iters_each_epoch=len(train_loader), last_epoch=last_epoch)


    """adapt to distributed training"""
    if cfg.OPT.DIST_TRAIN:
        denoiser = DistributedDataParallel(denoiser, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()], find_unused_parameters=True)


    """start training"""
    logger.info('**********************Start training %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    train_model(denoiser, optimizer, scheduler, train_loader, ema_helper, cfg,
                start_epoch, it, logger, wb_log, 
                train_sampler=train_sampler, test_loader=test_loader,
                ckpt_save_interval=args.ckpt_save_interval, ckpt_save_time_interval=args.ckpt_save_time_interval,
                max_ckpt_save_num=args.max_ckpt_save_num, logger_iter_interval=args.logger_iter_interval)

    logger.info('**********************End training %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


    """start evaluation"""
    logger.info('**********************Start evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))

    # tweak eval settings#
    args.start_epoch = max(args.epochs, 0)      # Only evaluate the last 10 epochs
    cfg.DATA_CONFIG.SAMPLE_INTERVAL.val = 1     # Evaluate all samples
    args.interactive = False                    # do not run interactive evaluation
    args.submit = False                         # do not generate submission files
    repeat_eval_ckpt(denoiser.module if cfg.OPT.DIST_TRAIN else denoiser, test_loader, cfg, args, logger, 
                     args_ema_coef=None)

    logger.info('**********************End evaluation %s/%s(%s)**********************' %
                (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()

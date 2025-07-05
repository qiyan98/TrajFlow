# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from trajflow.datasets import build_dataloader


def init_dataloader(cfg, logger):
    # Use per-GPU batch size if in distributed mode, otherwise use total batch size.
    if cfg.OPT.DIST_TRAIN:
        train_batch_size = cfg.OPT.BATCH_SIZE_PER_GPU
    else:
        train_batch_size = cfg.OPT.TOTAL_GPUS * cfg.OPT.BATCH_SIZE_PER_GPU

    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=train_batch_size,
        dist=cfg.OPT.DIST_TRAIN, workers=cfg.OPT.WORKERS,
        logger=logger, training=True, testing=False, inter_pred=False)

    if cfg.OPT.DIST_TRAIN:
        test_batch_size = cfg.OPT.BATCH_SIZE_PER_GPU * 4
    else:
        test_batch_size = train_batch_size * 2  # or adjust as needed for non-DDP

    test_set, test_loader, test_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, batch_size=test_batch_size,
        dist=cfg.OPT.DIST_TRAIN, workers=cfg.OPT.WORKERS, 
        logger=logger, training=False, testing=False, inter_pred=False)

    return train_set, train_loader, train_sampler, test_set, test_loader, test_sampler

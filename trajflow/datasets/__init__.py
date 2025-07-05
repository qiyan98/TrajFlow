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


import torch
from torch.utils.data import DataLoader
from trajflow.utils import common_utils

from .waymo.waymo_dataset import WaymoDataset


__all__ = {
    'WaymoDataset': WaymoDataset,
}


def build_dataloader(dataset_cfg, batch_size, dist, workers=4,
                     logger=None, 
                     training=True, testing=False, inter_pred=False):

    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        training=training,
        testing=testing,
        inter_pred=inter_pred,
        logger=logger, 
    )

    if dist:
        if training:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            rank, world_size = common_utils.get_dist_info()
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, world_size, rank, shuffle=False)
    else:
        sampler = None

    drop_last = dataset_cfg.get('DATALOADER_DROP_LAST', False) and training
    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=(sampler is None) and training, collate_fn=dataset.collate_batch,
        drop_last=drop_last, sampler=sampler, timeout=0, 
    )

    return dataset, dataloader, sampler

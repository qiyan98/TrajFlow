# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch


META_INFO = {
    # 'range_x': [-75, 300],      # from val GT
    # 'range_y': [-120, 125],     # from val GT
    # 'range_x': [-240, 335],     # from train GT
    # 'range_y': [-130, 135],     # from train GT
    # 'range_x': [-250.0, 350.0],       # overall GT
    # 'range_y': [-150.0, 150.0],       # overall GT
    'range_x': [-10.0, 170.0],      # 0.1% - 99.9% percentile
    'range_y': [-60.0, 60.0],       # 0.1% - 99.9% percentile
    # 'range_x': [0.0, 2.0],  # hack, not really changing the range
    # 'range_y': [0.0, 2.0],  # hack, not really changing the range

    'sqrt_x_coef': 15.0,
    'sqrt_y_coef': 15.0,
    'cbrt_x_coef': 8.0,
    'cbrt_y_coef': 8.0,

    'sqrt_x_offset': -0.3,
    'sqrt_y_offset': 0.0,
}


def shift_data_to_normalize(traj_data, traj_mask, data_rescale='sqrt', meta_info=META_INFO):
    """
    @param traj_data: [N, T, 2] or [N, T, 3] or [N, X, Y, 3]
    @param tarj_mask: [N, T]
    @param meta_info: dict
    @param data_rescale: str
    """
    traj_data = traj_data.clone()
    assert traj_data.size(-1) == 2 or traj_data.size(-1) == 3  # the third dimension for z-coord is not changed if it exists

    if traj_data.size(-1) == 3:
        traj_data_z = traj_data[..., 2].clone()
    if traj_mask is None:
        traj_mask = torch.ones_like(traj_data[..., 0]).bool()
    else:
        if len(traj_mask.shape) in [len(traj_data.shape) - 1, len(traj_data.shape)]:
            pass
        else:
            breakpoint()
    ori_padded_data = traj_data[torch.logical_not(traj_mask)]

    if data_rescale == 'linear':
        min_x, max_x = meta_info['range_x']
        min_y, max_y = meta_info['range_y']

        traj_data[..., 0] = (traj_data[..., 0] - min_x) / (max_x - min_x) * 2 - 1
        traj_data[..., 1] = (traj_data[..., 1] - min_y) / (max_y - min_y) * 2 - 1
    elif data_rescale == 'sqrt':
        traj_data = torch.abs(traj_data).sqrt() * torch.sign(traj_data)  # [N, T, 2]
        traj_data[..., 0] = traj_data[..., 0] / meta_info['sqrt_x_coef'] + meta_info['sqrt_x_offset']    # x-coord
        traj_data[..., 1] = traj_data[..., 1] / meta_info['sqrt_y_coef'] + meta_info['sqrt_y_offset']    # y-coord
    elif data_rescale == 'cbrt':
        traj_data = torch.abs(traj_data).pow(1/3) * torch.sign(traj_data)
        traj_data[..., 0] = traj_data[..., 0] / meta_info['cbrt_x_coef']    # x-coord
        traj_data[..., 1] = traj_data[..., 1] / meta_info['cbrt_y_coef']    # y-coord
    elif data_rescale == 'log_center':
        # center the data around 0, then take log
        min_x, max_x = meta_info['range_x']
        min_y, max_y = meta_info['range_y']
        traj_data[..., 0] = traj_data[..., 0] - (min_x + max_x) / 2  # de-mean x
        traj_data[..., 1] = traj_data[..., 1] - (min_y + max_y) / 2  # de-mean y

        traj_data = torch.log(traj_data.abs() + 1) * torch.sign(traj_data)
        traj_data[..., 0] /= np.log(max_x - (min_x + max_x) / 2 + 1)
        traj_data[..., 1] /= np.log(max_y - (min_y + max_y) / 2 + 1)

    if traj_data.size(-1) == 3:
        traj_data[..., 2] = traj_data_z
    traj_data[torch.logical_not(traj_mask)] = ori_padded_data
    return traj_data


def shift_data_to_denormalize(traj_data, traj_mask, data_rescale='sqrt', meta_info=META_INFO):
    """
    @param traj_data: [N, T, 2]
    @param tarj_mask: [N, T]
    @param meta_info: dict
    @param data_rescale: str
    """
    traj_data = traj_data.clone()
    assert traj_data.size(-1) == 2
    if traj_mask is None:
        traj_mask = torch.ones_like(traj_data[..., 0]).bool()
    else:
        traj_mask = traj_mask.unsqueeze(1).expand(-1, traj_data.size(1), -1) if len(traj_data.shape) == 4 else traj_mask

    flag_apply_mask = torch.logical_not(traj_mask).sum()
    if flag_apply_mask:
        ori_pad_val = traj_data[torch.logical_not(traj_mask)].unique()
        # if len(ori_pad_val) != 1:
        #     breakpoint()
        assert len(ori_pad_val) == 1
        ori_pad_val = ori_pad_val[0]

    if data_rescale == 'linear':
        min_x, max_x = meta_info['range_x']
        min_y, max_y = meta_info['range_y']
        traj_data[..., 0] = (traj_data[..., 0] + 1) / 2 * (max_x - min_x) + min_x
        traj_data[..., 1] = (traj_data[..., 1] + 1) / 2 * (max_y - min_y) + min_y
    elif data_rescale == 'sqrt':
        traj_data[..., 0] = (traj_data[..., 0] - meta_info['sqrt_x_offset']) * meta_info['sqrt_x_coef']
        traj_data[..., 1] = (traj_data[..., 1] - meta_info['sqrt_y_offset']) * meta_info['sqrt_y_coef']
        traj_data = traj_data.abs().pow(2) * traj_data.sign() 
    elif data_rescale == 'cbrt':
        traj_data[..., 0] = traj_data[..., 0] * meta_info['cbrt_x_coef']
        traj_data[..., 1] = traj_data[..., 1] * meta_info['cbrt_y_coef']
        traj_data = traj_data.abs().pow(3) * traj_data.sign()
    elif data_rescale == 'log_center':
        min_x, max_x = meta_info['range_x']
        min_y, max_y = meta_info['range_y']
        traj_data[..., 0] = traj_data[..., 0] * np.log(max_x - (min_x + max_x) / 2 + 1)
        traj_data[..., 1] = traj_data[..., 1] * np.log(max_y - (min_y + max_y) / 2 + 1)

        traj_data = (torch.exp(traj_data.abs()) - 1) * traj_data.sign()
        traj_data[..., 0] += (min_x + max_x) / 2
        traj_data[..., 1] += (min_y + max_y) / 2

    if flag_apply_mask:
        traj_data[torch.logical_not(traj_mask)] = ori_pad_val

    return traj_data


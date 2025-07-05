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


import numpy as np
import torch
import torch.nn.functional as F


from trajflow.utils.mtr_loss_utils import nll_loss_gmm_direct


class LossBuffer:
    def __init__(self, t_min, t_max, num_time_steps):
        """
        Initialize the LossBuffer with the specified number of denoising levels.
        """
        self.t_min = t_min
        self.t_max = t_max
        self.num_time_steps = num_time_steps
        self.t_interval = np.linspace(t_min, t_max, num_time_steps)
        self.loss_data = [[] for _ in range(self.num_time_steps)]
        self.last_epoch = -1

    def record_loss(self, t, loss, epoch_id):
        """
        Record the loss for a specific denoising level.
        @param t:       [B] the denoising level.
        @param loss:    [B] the loss value.    
        """

        flag_reset = False
        if epoch_id != self.last_epoch:
            self.last_epoch = epoch_id
            self.reset()
            flag_reset = epoch_id > 0
        
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        if isinstance(loss, torch.Tensor):
            loss = loss.cpu().numpy()
        
        idx = np.digitize(t, self.t_interval) - 1
        for i, l in zip(idx, loss):
            self.loss_data[i].append(l)

        return flag_reset

    def reset(self):
        """
        Reset the loss data for a new epoch.
        """
        self.loss_data = [[] for _ in range(self.num_time_steps)]

    def get_average_loss(self):
        """
        To be used for plotting a histogram of denoising level vs. average loss for the last epoch.
        """
        avg_loss_per_level = [np.mean(l) if len(l) > 0 else -1.0 for l in self.loss_data]
        dict_loss_per_level = {t: l for t, l in zip(self.t_interval, avg_loss_per_level)}
        return dict_loss_per_level


def first_occurrence_mask_fast(x):
    # x: [B, N]
    B, N = x.size()
    mask = torch.zeros_like(x, dtype=torch.bool)
    for i in range(B):
        # torch.unique with sorted=False preserves the order of appearance.
        _, first_indices = torch.unique(x[i], sorted=False, return_inverse=True)
        mask[i, first_indices] = True
    return mask


def plackett_luce_loss(logits, preference_argsort):
    """
    Compute the Plackett-Luce loss for a batch of samples.
    @params logits:                 [B, N], predicted logits (unnormalized scores) for each item
    @params preference_argsort:     [B, N], the preference order of the items (from the best to the worst)
    Note: ranks_idx must be distinct and in the range [0, N-1]
    """

    # Reorder logits according to ranks, from the best to the worst
    # z[r_1], z[r_2], ..., z[r_N], level of preference: r_1 > r_2 > ... > r_N
    ordered_logits = torch.gather(logits, dim=1, index=preference_argsort)          # [B, N]

    # Compute cumulative log-sum-exp
    cumulative_log_sum_exp = torch.logcumsumexp(ordered_logits, dim=-1)             # [B, N]

    # Compute the loss
    log_probs = ordered_logits - cumulative_log_sum_exp                             # z[r_i] - log(sum(exp(z[r_i:]))) for all i
    loss = -log_probs

    # Check the uniqueness of the ranks, ignore repeated ranks
    loss_mask = first_occurrence_mask_fast(preference_argsort)                      # [B, N]
    loss = (loss * loss_mask.float()).mean(dim=-1)                                  # [B]
    return loss


def get_dense_future_prediction_loss(forward_ret_dict, wb_pre_tag='', wb_dict=None, disp_dict=None):
    obj_trajs_future_state = forward_ret_dict['obj_trajs_future_state'].cuda()
    obj_trajs_future_mask = forward_ret_dict['obj_trajs_future_mask'].cuda()
    pred_dense_trajs = forward_ret_dict['pred_dense_trajs'] 
    assert pred_dense_trajs.shape[-1] == 7
    assert obj_trajs_future_state.shape[-1] == 4

    pred_dense_trajs_gmm, pred_dense_trajs_vel = pred_dense_trajs[:, :, :, 0:5], pred_dense_trajs[:, :, :, 5:7]

    loss_reg_vel = F.l1_loss(pred_dense_trajs_vel, obj_trajs_future_state[:, :, :, 2:4], reduction='none')
    loss_reg_vel = (loss_reg_vel * obj_trajs_future_mask[:, :, :, None]).sum(dim=-1).sum(dim=-1)

    num_center_objects, num_objects, num_timestamps, _ = pred_dense_trajs.shape
    fake_scores = pred_dense_trajs.new_zeros((num_center_objects, num_objects)).view(-1, 1)  # (num_center_objects * num_objects, 1)

    temp_pred_trajs = pred_dense_trajs_gmm.contiguous().view(num_center_objects * num_objects, 1, num_timestamps, 5)
    temp_gt_idx = torch.zeros(num_center_objects * num_objects).cuda().long()  # (num_center_objects * num_objects)
    temp_gt_trajs = obj_trajs_future_state[:, :, :, 0:2].contiguous().view(num_center_objects * num_objects, num_timestamps, 2)
    temp_gt_trajs_mask = obj_trajs_future_mask.view(num_center_objects * num_objects, num_timestamps)
    loss_reg_gmm, _ = nll_loss_gmm_direct(
        pred_scores=fake_scores, pred_trajs=temp_pred_trajs, gt_trajs=temp_gt_trajs, gt_valid_mask=temp_gt_trajs_mask,
        pre_nearest_mode_idxs=temp_gt_idx,
        timestamp_loss_weight=None, use_square_gmm=False,
    )
    loss_reg_gmm = loss_reg_gmm.view(num_center_objects, num_objects)

    loss_reg = loss_reg_vel + loss_reg_gmm

    obj_valid_mask = obj_trajs_future_mask.sum(dim=-1) > 0

    loss_reg = (loss_reg * obj_valid_mask.float()).sum(dim=-1) / torch.clamp_min(obj_valid_mask.sum(dim=-1), min=1.0)
    loss_reg = loss_reg.mean()

    if wb_dict is None:
        wb_dict = {}
    if disp_dict is None:
        disp_dict = {}

    wb_dict[f'{wb_pre_tag}loss_dense_prediction'] = loss_reg.item()
    return loss_reg, wb_dict, disp_dict

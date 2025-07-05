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

from trajflow.models.layers.common_layers import gen_sineembed_for_position
from trajflow.utils import motion_utils


def apply_dense_future_prediction(obj_feature, obj_mask, obj_pos, forward_ret_dict,
                                  obj_pos_encoding_layer, dense_future_head, future_traj_mlps, traj_fusion_mlps, 
                                  num_future_frames):
    num_center_objects, num_objects, _ = obj_feature.shape

    # dense future prediction
    obj_pos_valid = obj_pos[obj_mask][..., 0:2]
    obj_feature_valid = obj_feature[obj_mask]
    obj_pos_feature_valid = obj_pos_encoding_layer(obj_pos_valid)
    obj_fused_feature_valid = torch.cat((obj_pos_feature_valid, obj_feature_valid), dim=-1)

    pred_dense_trajs_valid = dense_future_head(obj_fused_feature_valid)
    pred_dense_trajs_valid = pred_dense_trajs_valid.view(pred_dense_trajs_valid.shape[0], 
            num_future_frames, 7)

    temp_center = pred_dense_trajs_valid[:, :, 0:2] + obj_pos_valid[:, None, 0:2]
    pred_dense_trajs_valid = torch.cat((temp_center, pred_dense_trajs_valid[:, :, 2:]), dim=-1)

    # future feature encoding and fuse to past obj_feature
    obj_future_input_valid = pred_dense_trajs_valid[:, :, [0, 1, -2, -1]].flatten(start_dim=1, end_dim=2) 
    obj_future_feature_valid = future_traj_mlps(obj_future_input_valid)

    obj_full_trajs_feature = torch.cat((obj_feature_valid, obj_future_feature_valid), dim=-1)
    obj_feature_valid = traj_fusion_mlps(obj_full_trajs_feature)

    ret_obj_feature = torch.zeros_like(obj_feature)
    ret_obj_feature[obj_mask] = obj_feature_valid

    ret_pred_dense_future_trajs = obj_feature.new_zeros(num_center_objects, num_objects, num_future_frames, 7)
    ret_pred_dense_future_trajs[obj_mask] = pred_dense_trajs_valid
    forward_ret_dict['pred_dense_trajs'] = ret_pred_dense_future_trajs

    return ret_obj_feature, ret_pred_dense_future_trajs
    

def get_motion_query(intention_points_dict, intention_query_mlps, center_objects_type):
    num_center_objects = len(center_objects_type)
    intention_points = torch.stack([intention_points_dict[center_objects_type[obj_idx]] for obj_idx in range(num_center_objects)], dim=0)  # [B, K, 2]
    d_model = intention_query_mlps[0].in_features
    intention_query = gen_sineembed_for_position(intention_points, hidden_dim=d_model)  # [B, K, D]
    intention_query = intention_query_mlps(intention_query.view(-1, d_model)).view(num_center_objects, -1, d_model)  # [B, K, D]
    return intention_query, intention_points
    

def get_center_gt_idx(layer_idx, num_inter_layers, num_decoder_layers, flag_training, forward_ret_dict, 
                      pred_scores=None, pred_trajs=None, pred_list=None, prev_trajs=None, prev_dist=None):
    if flag_training:
        center_gt_trajs = forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = forward_ret_dict['center_gt_trajs_mask'].cuda()
        center_gt_final_valid_idx = forward_ret_dict['center_gt_final_valid_idx'].long()
        intention_points = forward_ret_dict['intention_points']
        num_center_objects = center_gt_trajs.shape[0]

        center_gt_goals = center_gt_trajs[torch.arange(num_center_objects), center_gt_final_valid_idx, 0:2] 
        if (layer_idx // num_inter_layers) * num_inter_layers - 1 < 0:
            dist = (center_gt_goals[:, None, :] - intention_points).norm(dim=-1)  # (num_center_objects, num_query)
            anchor_trajs = intention_points.unsqueeze(-2)
            select_mask = None
            select_idx = None
            if pred_list is None:
                center_gt_positive_idx = dist.argmin(dim=-1)  # (num_center_objects)
                return center_gt_positive_idx, anchor_trajs, dist, select_mask, select_idx

            center_gt_positive_idx, select_mask, select_idx = motion_utils.select_distinct_anchors(
                dist, pred_scores, pred_trajs, anchor_trajs
            )
            return center_gt_positive_idx, anchor_trajs, dist, select_mask, select_idx

        # Evolving & Distinct Anchors
        if pred_list is None:
            unique_layers = set(
                [(i//num_inter_layers)* num_inter_layers
                    for i in range(num_decoder_layers)]
            )
            if layer_idx in unique_layers:
                anchor_trajs = pred_trajs
                dist = ((center_gt_trajs[:, None, :, 0:2] - anchor_trajs[..., 0:2]).norm(dim=-1) * \
                        center_gt_trajs_mask[:, None]).sum(dim=-1) 
            else:
                anchor_trajs, dist = prev_trajs, prev_dist
        else:
            anchor_trajs, dist = motion_utils.get_evolving_anchors(
                layer_idx, num_inter_layers, pred_list, 
                center_gt_goals, intention_points, 
                center_gt_trajs, center_gt_trajs_mask, 
                )

        center_gt_positive_idx, select_mask, select_idx = motion_utils.select_distinct_anchors(
            dist, pred_scores, pred_trajs, anchor_trajs
        )
    else:
        center_gt_positive_idx = None
        anchor_trajs, dist = None, None
        select_mask=None
        select_idx=None
        
    return center_gt_positive_idx, anchor_trajs, dist, select_mask, select_idx
    

def apply_cross_attention(query_feat, kv_feat, kv_mask,
                          query_pos_feat, kv_pos_feat, 
                          pred_query_center, attn_indexing,
                          attention_layer,
                          query_feat_pre_mlp=None, query_embed_mlp=None,
                          query_feat_pos_mlp=None, is_first=False
                          ):
    """
    Args:
        query_feat, query_pos_feat, query_searching_feat  [M, B, D]
        kv_feat, kv_pos_feat  [B, N, D]
        kv_mask [B, N]
        attn_indexing [B, N, M]
        attention_layer (func): LocalTransformer Layer (as in EQNet and MTR)
        query_feat_pre_mlp, query_embed_mlp, query_feat_pos_mlp (nn.Linear):
        projections to align decoder dimension
        is_first (bool): whether to concat query pos feature (as in MTR) 
    Returns:
        query_feat: (B, M, D)
    """

    if query_feat_pre_mlp is not None:
        query_feat = query_feat_pre_mlp(query_feat)
    if query_embed_mlp is not None:
        query_pos_feat = query_embed_mlp(query_pos_feat)
    
    d_model = query_feat.shape[-1]
    query_searching_feat = gen_sineembed_for_position(pred_query_center, hidden_dim=d_model)
    
    # fast attention
    if attn_indexing is not None:
        B, K = attn_indexing.shape[:2]
        M = kv_mask.shape[-1]
        context_valid_mask_ = torch.zeros([B, K, M+1], dtype=torch.bool, device=kv_mask.device)
        context_valid_mask_.scatter_(2, (attn_indexing + 1).long(), torch.ones_like(attn_indexing).bool())
        context_valid_mask = torch.logical_and(context_valid_mask_[:, :, 1:], kv_mask[:, None, :])  # [B, K, M]
    else:
        context_valid_mask = kv_mask

    # batch-major tensor shape
    query_feat = attention_layer(
        query=query_feat,
        context=kv_feat,
        context_valid_mask=context_valid_mask,
        query_sa_pos_embeddings=query_pos_feat,
        query_ca_pos_embeddings=query_searching_feat,
        context_ca_pos_embeddings=kv_pos_feat,
        is_first=is_first,
        context_indexing=attn_indexing
        )  # [B, M, D] 

    if query_feat_pos_mlp is not None:
        query_feat = query_feat_pos_mlp(query_feat)

    return query_feat


def generate_final_prediction(pred_list, num_motion_modes):
    pred_scores, pred_trajs = pred_list[-1][:2]
    pred_scores = torch.sigmoid(pred_scores)
    
    num_query = pred_trajs.shape[1]

    if num_motion_modes != num_query:
        assert num_query > num_motion_modes
        pred_trajs_final, pred_scores_final, selected_idxs = motion_utils.inference_distance_nms(
            pred_scores, pred_trajs, num_motion_modes)
    else:
        pred_trajs_final = pred_trajs
        pred_scores_final = pred_scores
        selected_idxs = None

    return pred_scores_final, pred_trajs_final, selected_idxs

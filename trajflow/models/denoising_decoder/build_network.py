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


import copy
import pickle
import torch
import torch.nn as nn


from trajflow.models.layers.common_layers import build_mlps
from trajflow.config import init_cfg
from trajflow.models.layers.transformer.dmt_decoder_layer import DMTDecoderLayer


def build_in_proj_layer(d_input, d_model, d_obj, d_map):
    in_proj_center_obj = build_mlps(c_in=d_input, mlp_channels=[d_model] * 2, ret_before_act=True, without_norm=True)
    in_proj_obj = build_mlps(c_in=d_input, mlp_channels=[d_obj] * 2, ret_before_act=True, without_norm=True)
    in_proj_map = build_mlps(c_in=d_input, mlp_channels=[d_map] * 2, ret_before_act=True, without_norm=True)
    return in_proj_center_obj, in_proj_obj, in_proj_map


def build_transformer_decoder(d_tf, nhead, dropout, num_decoder_layers):
    decoder_layer = DMTDecoderLayer(d_model=d_tf, nhead=nhead, dim_feedforward=d_tf * 4, 
                                    dropout=dropout, activation="relu", normalize_before=False,
                                    use_concat_pe_ca=True, normalization_type='layer_norm', bias=True, 
                                    qk_norm=False, adaLN=False)
    decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
    return decoder_layers


def build_dense_future_prediction_layers(hidden_dim, d_obj, num_future_frames):
    obj_pos_encoding_layer = build_mlps(c_in=2, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True)
    dense_future_head = build_mlps(c_in=hidden_dim + d_obj, mlp_channels=[hidden_dim, hidden_dim, num_future_frames * 7], ret_before_act=True)
    future_traj_mlps = build_mlps(c_in=4 * num_future_frames, mlp_channels=[hidden_dim, hidden_dim, hidden_dim], ret_before_act=True, without_norm=True)
    traj_fusion_mlps = build_mlps(c_in=hidden_dim + d_obj, mlp_channels=[hidden_dim, hidden_dim, d_obj], ret_before_act=True, without_norm=True)
    return obj_pos_encoding_layer, dense_future_head, future_traj_mlps, traj_fusion_mlps
        

def build_motion_query(d_model, model_cfg):
    _init_cfg = init_cfg()
    intention_points_file = _init_cfg.ROOT_DIR / model_cfg.INTENTION_POINTS_FILE
    with open(intention_points_file, 'rb') as f:
        intention_points_dict = pickle.load(f)
    intention_points = {}
    for cur_type in model_cfg.OBJECT_TYPE:
        cur_intention_points = intention_points_dict[cur_type]
        cur_intention_points = torch.from_numpy(cur_intention_points).float().view(-1, 2).cuda()
        intention_points[cur_type] = cur_intention_points

    intention_query_mlps = build_mlps(c_in=d_model, mlp_channels=[d_model, d_model], ret_before_act=True)
    return intention_points, intention_query_mlps


def build_motion_head(d_model, map_d_model, hidden_size, num_future_frames, num_decoder_layers):
    temp_layer = build_mlps(c_in=d_model * 2 + map_d_model, mlp_channels=[d_model, d_model], ret_before_act=True)
    query_feature_fusion_layers = nn.ModuleList([copy.deepcopy(temp_layer) for _ in range(num_decoder_layers)])

    motion_reg_head =  build_mlps(c_in=d_model, mlp_channels=[hidden_size, hidden_size, num_future_frames * 7], ret_before_act=True)
    motion_cls_head =  build_mlps(c_in=d_model, mlp_channels=[hidden_size, hidden_size, 1], ret_before_act=True)

    motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
    motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])
    return query_feature_fusion_layers, motion_reg_heads, motion_cls_heads
    

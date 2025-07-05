# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from trajflow.utils.denoising_data_rescale import shift_data_to_normalize
from trajflow.models.layers.common_layers import build_mlps, gen_sineembed_for_position, TimestepEmbedder
from trajflow.utils.common_utils import apply_dynamic_map_collection
from trajflow.utils.mtr_loss_utils import nll_loss_gmm_direct
from trajflow.utils import motion_utils
from trajflow.utils.common_utils import register_module_to_params_dict

from trajflow.models.denoising_decoder.build_network import build_in_proj_layer, build_transformer_decoder, build_dense_future_prediction_layers, build_motion_query, build_motion_head
from trajflow.models.denoising_decoder.decoder_utils import get_motion_query, get_center_gt_idx, apply_dense_future_prediction, apply_cross_attention, generate_final_prediction
from trajflow.models.denoising_decoder.compute_loss import LossBuffer, plackett_luce_loss, get_dense_future_prediction_loss


from einops import rearrange


class DenoisingDecoder(nn.Module):
    def __init__(self, model_cfg, denoising_cfg, logger, save_dirs, data_rescale):
        super().__init__()

        self.model_cfg = model_cfg
        self.denoising_cfg = denoising_cfg
        self.logger = logger
        self.save_dirs = save_dirs
        self.data_rescale = data_rescale

        self.object_type = self.model_cfg.OBJECT_TYPE
        self.num_future_frames = self.model_cfg.NUM_FUTURE_FRAMES
        self.num_motion_modes = self.model_cfg.NUM_MOTION_MODES
        self.d_model = self.model_cfg.D_QUERY
        self.num_decoder_layers = self.model_cfg.DEPTH

        self.num_inter_layers = 2

        self.params_dict = {}
        self.register_module_to_params_dict = lambda module, name: register_module_to_params_dict(self.params_dict, module, name)

        self.build_denoising_layers()

        # Denoising loss history buffer
        self.ctc_loss = self.denoising_cfg.get('CTC_LOSS', False)
        if self.ctc_loss:
            self.loss_buffer_ctc_1 = LossBuffer(t_min=0.0, t_max=1.0, num_time_steps=100)
            self.loss_buffer_ctc_2 = LossBuffer(t_min=0.0, t_max=1.0, num_time_steps=100)
        else:
            self.loss_buffer = LossBuffer(t_min=0.0, t_max=1.0, num_time_steps=100)

    def build_denoising_layers(self):
        """
        Building the denoising decoder layers.
        """

        """Denoising token and time embedding layers"""        
        self.x_embedder = build_mlps(c_in=2 * self.num_future_frames, mlp_channels=[self.d_model] * 2, ret_before_act=True, without_norm=True)
        self.t_embedder = TimestepEmbedder(self.d_model)   
        self.feat_fusion_mlp = build_mlps(c_in=self.d_model * 3, mlp_channels=[self.d_model] * 2, ret_before_act=True, without_norm=False, layer_norm=True)
        _feat_tf_encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.model_cfg.HEADS, 
                                                      dim_feedforward=self.d_model * 4, dropout=self.model_cfg.DROPOUT, batch_first=True)
        self.feat_tf_encoder = nn.TransformerEncoder(_feat_tf_encoder, num_layers=2)
        
        self.register_module_to_params_dict(nn.ModuleList([self.x_embedder, self.t_embedder, self.feat_fusion_mlp, self.feat_tf_encoder]), 
                                            'denoising_layers')

        """Plackett-Luce loss layers"""
        self.pl_readout_mlp = nn.ModuleList([build_mlps(c_in=self.d_model, mlp_channels=[self.d_model, self.d_model, 1], 
                                                        ret_before_act=True, without_norm=False, layer_norm=True) for _ in range(self.num_decoder_layers)])
        self.register_module_to_params_dict(nn.ModuleList([self.pl_readout_mlp]), 'pl_readout_layers')

        """Query-based cross-attention layers"""
        # breakpoint()
        d_input = self.model_cfg.CONTEXT_D_MODEL
        d_obj = self.model_cfg.D_OBJ
        d_map = self.model_cfg.D_MAP

        # encoder token projection layers
        self.in_proj_center_obj, self.in_proj_obj, self.in_proj_map = build_in_proj_layer(d_input, self.d_model, d_obj, d_map)

        # query-to-object cross-attention layers
        self.obj_decoder_layers = build_transformer_decoder(d_tf=d_obj, nhead=self.model_cfg.HEADS, 
                                                            dropout=self.model_cfg.DROPOUT, num_decoder_layers=self.num_decoder_layers)

        # query-to-map cross-attention layers
        self.map_decoder_layers = build_transformer_decoder(d_tf=d_map, nhead=self.model_cfg.HEADS, 
                                                            dropout=self.model_cfg.DROPOUT, num_decoder_layers=self.num_decoder_layers)

        # build MLPs for decoder token dimension alignment
        self.actor_query_content_mlps = nn.ModuleList([copy.deepcopy(nn.Linear(self.d_model, d_obj)) for _ in range(self.num_decoder_layers)])
        self.actor_query_content_mlps_reverse = nn.ModuleList([copy.deepcopy(nn.Linear(d_obj, self.d_model)) for _ in range(self.num_decoder_layers)])
        self.actor_query_embed_mlps = nn.Linear(self.d_model, d_obj)

        self.map_query_content_mlps = nn.ModuleList([copy.deepcopy(nn.Linear(self.d_model, d_map)) for _ in range(self.num_decoder_layers)])
        self.map_query_embed_mlps = nn.Linear(self.d_model, d_map)

        self.register_module_to_params_dict(nn.ModuleList([self.actor_query_content_mlps, self.actor_query_content_mlps_reverse, self.actor_query_embed_mlps]), 'mlp_obj_query')   
        self.register_module_to_params_dict(nn.ModuleList([self.map_query_content_mlps, self.map_query_embed_mlps]), 'mlp_map_query')
        self.register_module_to_params_dict(nn.ModuleList([self.in_proj_center_obj, self.in_proj_obj, self.in_proj_map]), 'mlp_in_proj')
        self.register_module_to_params_dict(self.obj_decoder_layers, 'obj_tf_decoder')
        self.register_module_to_params_dict(self.map_decoder_layers, 'map_tf_decoder')

        """Dense future prediction layers"""
        self.obj_pos_encoding_layer, self.dense_future_head, self.future_traj_mlps, self.traj_fusion_mlps = build_dense_future_prediction_layers(
            hidden_dim=self.d_model, d_obj=d_obj, num_future_frames=self.num_future_frames)
        self.register_module_to_params_dict(nn.ModuleList([self.obj_pos_encoding_layer, self.dense_future_head, self.future_traj_mlps, self.traj_fusion_mlps]), 'mlp_dense_pred')

        """Motion query layers"""
        self.intention_points, self.intention_query_mlps = build_motion_query(self.d_model, self.model_cfg)
        self.register_module_to_params_dict(nn.ModuleList([self.intention_query_mlps]), 'mlp_motion_query')

        """Motion head layers"""
        self.query_feature_fusion_layers, self.motion_reg_heads, self.motion_cls_heads = build_motion_head(
            d_model=self.d_model, map_d_model=d_map, hidden_size=self.d_model, 
            num_future_frames=self.num_future_frames, num_decoder_layers=self.num_decoder_layers)

        self.register_module_to_params_dict(self.query_feature_fusion_layers, 'mlp_query_feature_fusion')
        self.register_module_to_params_dict(nn.ModuleList([self.motion_reg_heads, self.motion_cls_heads]), 'mlp_motion_head')

        """Forward return dict cache"""
        self.forward_ret_dict = {}

        """Parameter breakdown"""
        params_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        params_other = params_total - sum(self.params_dict.values())
        self.params_dict = {'total': params_total, **self.params_dict}
        self.params_dict = {**self.params_dict, 'other': params_other}
        self.logger.info("===== DenoisingDecoder parameters breakdown =====")
        for nm, p in self.params_dict.items():
            self.logger.info("#params for {:40}: {:,}".format(nm, p))
        self.logger.info("===== DenoisingDecoder parameters breakdown =====")

    def apply_transformer_decoder(self, center_objects_feature, center_objects_type,
                                  obj_feature, obj_mask, obj_pos, 
                                  map_feature, map_mask, map_pos):
        # get motion query
        intention_query, intention_points = get_motion_query(self.intention_points, self.intention_query_mlps, center_objects_type)
        self.forward_ret_dict['intention_points'] = intention_points    # [B, K, 2]
        num_center_objects = intention_query.shape[0]
        num_query = intention_query.shape[1]

        # init query content, pred waypoints, and dynamic query center
        query_content = torch.zeros_like(intention_query)               # [B, K, D]
        pred_waypoints = intention_points[:, :, None, :]                # [B, K, 1, 2]
        dynamic_query_center = intention_points                         # [B, K, 2]

        # init anchor trajs and dist
        base_map_idxs = None
        pred_scores, pred_trajs = None, None
        anchor_trajs, anchor_dist = None, None
        pred_list = []

        # get map and obj pos embed
        map_pos_embed = gen_sineembed_for_position(map_pos[:, :, 0:2], hidden_dim=map_feature.shape[-1])
        obj_pos_embed = gen_sineembed_for_position(obj_pos[:, :, 0:2], hidden_dim=obj_feature.shape[-1])

        for layer_idx in range(self.num_decoder_layers):
            # get anchor trajs and dist
            _, anchor_trajs, anchor_dist, _, _ = get_center_gt_idx(
                layer_idx, self.num_inter_layers, self.num_decoder_layers, self.training, self.forward_ret_dict, 
                pred_scores, pred_trajs, prev_trajs=anchor_trajs, prev_dist=anchor_dist)

            # apply dynamic map indexing
            map_attn_idxs, base_map_idxs = apply_dynamic_map_collection(
                map_pos=map_pos, map_mask=map_mask,
                pred_waypoints=pred_waypoints,
                base_region_offset=self.model_cfg.CENTER_OFFSET_OF_MAP,
                num_waypoint_polylines=self.model_cfg.NUM_WAYPOINT_MAP_POLYLINES,
                num_base_polylines=self.model_cfg.NUM_BASE_MAP_POLYLINES,
                base_map_idxs=base_map_idxs,
                num_query=num_query)

            # apply query-to-object cross-attention
            agent_query_feature = apply_cross_attention(
                query_feat=query_content, kv_feat=obj_feature, kv_mask=obj_mask,
                query_pos_feat=intention_query, kv_pos_feat=obj_pos_embed, 
                pred_query_center=dynamic_query_center, attn_indexing=None,
                attention_layer=self.obj_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.actor_query_content_mlps[layer_idx],
                query_embed_mlp=self.actor_query_embed_mlps,
                query_feat_pos_mlp=self.actor_query_content_mlps_reverse[layer_idx],
                is_first=layer_idx==0) 

            # apply query-to-map cross-attention
            map_query_feature = apply_cross_attention(
                query_feat=query_content, kv_feat=map_feature, kv_mask=map_mask,
                query_pos_feat=intention_query, kv_pos_feat=map_pos_embed, 
                pred_query_center=dynamic_query_center, attn_indexing=map_attn_idxs,
                attention_layer=self.map_decoder_layers[layer_idx],
                query_feat_pre_mlp=self.map_query_content_mlps[layer_idx],
                query_embed_mlp=self.map_query_embed_mlps,
                is_first=layer_idx==0)

            # prediction heads
            query_feature = torch.cat([center_objects_feature, agent_query_feature, map_query_feature], dim=-1)  # [B, K, 3D]
            query_content = self.query_feature_fusion_layers[layer_idx](
                query_feature.flatten(start_dim=0, end_dim=1)
                ).view(num_center_objects, num_query, -1)   # [B, K, D]

            pl_logits = self.pl_readout_mlp[layer_idx](query_content).squeeze(-1)    # [B, K]

            query_content_t = query_content.view(num_center_objects * num_query, -1)
            pred_scores = self.motion_cls_heads[layer_idx](query_content_t).view(num_center_objects, num_query)
            pred_trajs = self.motion_reg_heads[layer_idx](query_content_t).view(num_center_objects, num_query, self.num_future_frames, 7)

            pred_list.append([pred_scores, pred_trajs, pl_logits])
                
            # update pred waypoints and dynamic query center
            pred_waypoints = pred_trajs[:, :, :, 0:2].detach().clone()                      # [B, K, T, 2]
            dynamic_query_center = pred_trajs[:, :, -1, 0:2].detach().clone().contiguous()  # [B, K, 2]

        return pred_list
    
    def get_decoder_loss(self, wb_pre_tag=''):
        center_gt_trajs = self.forward_ret_dict['center_gt_trajs'].cuda()
        center_gt_trajs_mask = self.forward_ret_dict['center_gt_trajs_mask'].cuda()
        assert center_gt_trajs.shape[-1] == 4

        pred_list = self.forward_ret_dict['pred_list']

        num_center_objects = center_gt_trajs.shape[0]
        
        wb_dict = {}
        disp_dict = {}
        total_loss_b = 0
        total_loss_reg, total_loss_cls = 0, 0

        for layer_idx in range(self.num_decoder_layers):
            pred_scores, pred_trajs = pred_list[layer_idx][:2]
            
            assert pred_trajs.shape[-1] == 7
            pred_trajs_gmm, pred_vel = pred_trajs[:, :, :, 0:5], pred_trajs[:, :, :, 5:7]

            center_gt_positive_idx, _, _, select_mask, select_idx = get_center_gt_idx(
                layer_idx, self.num_inter_layers, self.num_decoder_layers, self.training, self.forward_ret_dict, 
                pred_scores, pred_trajs, pred_list)

            loss_reg_gmm, center_gt_positive_idx = nll_loss_gmm_direct(
                pred_scores=pred_scores, pred_trajs=pred_trajs_gmm,
                gt_trajs=center_gt_trajs[:, :, 0:2], gt_valid_mask=center_gt_trajs_mask,
                pre_nearest_mode_idxs=center_gt_positive_idx,
                timestamp_loss_weight=None, use_square_gmm=False,
            )

            pred_vel = pred_vel[torch.arange(num_center_objects), center_gt_positive_idx]
            loss_reg_vel = F.l1_loss(pred_vel, center_gt_trajs[:, :, 2:4], reduction='none')
            loss_reg_vel = (loss_reg_vel * center_gt_trajs_mask[:, :, None]).sum(dim=-1).sum(dim=-1)

            bce_target = torch.zeros_like(pred_scores)
            bce_target[torch.arange(num_center_objects), center_gt_positive_idx] = 1.0
            loss_cls = F.binary_cross_entropy_with_logits(input=pred_scores, target=bce_target, reduction='none')

            loss_cls = (loss_cls * select_mask).sum(dim=-1)

            # PL ranking loss, we re-use the select_idx to get the preference index
            pl_logits = pred_list[layer_idx][-1]  # [B, K]
            loss_pl = plackett_luce_loss(pl_logits, select_idx[:, :self.num_motion_modes])  # [B]

            # total loss
            weight_cls = self.model_cfg.LOSS_WEIGHTS.get('cls', 1.0)
            weight_reg = self.model_cfg.LOSS_WEIGHTS.get('reg', 1.0)
            weight_vel = self.model_cfg.LOSS_WEIGHTS.get('vel', 0.2)
            weight_pl = self.model_cfg.LOSS_WEIGHTS.get('pl', 0.1)

            layer_loss = loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel +\
                loss_cls.sum(dim=-1) * weight_cls + weight_pl * loss_pl
            
            total_loss_b += layer_loss

            total_loss_reg += (loss_reg_gmm * weight_reg + loss_reg_vel * weight_vel).mean()
            total_loss_cls += (loss_cls.sum(dim=-1) * weight_cls + weight_pl * loss_pl).mean()

            wb_dict[f'{wb_pre_tag}loss_layer{layer_idx}'] = layer_loss.mean().item()
            wb_dict[f'{wb_pre_tag}loss_layer{layer_idx}_reg_gmm'] = loss_reg_gmm.mean().item() * weight_reg
            wb_dict[f'{wb_pre_tag}loss_layer{layer_idx}_reg_vel'] = loss_reg_vel.mean().item() * weight_vel
            wb_dict[f'{wb_pre_tag}loss_layer{layer_idx}_cls'] = loss_cls.mean().item() * weight_cls
            wb_dict[f'{wb_pre_tag}loss_layer{layer_idx}_pl'] = loss_pl.mean().item() * weight_pl
   
            if layer_idx + 1 == self.num_decoder_layers:
                layer_wb_dict_ade = motion_utils.get_ade_of_each_category(
                    pred_trajs=pred_trajs_gmm[:, :, :, 0:2],
                    gt_trajs=center_gt_trajs[:, :, 0:2], gt_trajs_mask=center_gt_trajs_mask,
                    object_types=self.forward_ret_dict['center_objects_type'],
                    valid_type_list=self.object_type,
                    post_tag=f'_layer_{layer_idx}',
                    pre_tag=wb_pre_tag
                )
                wb_dict.update(layer_wb_dict_ade)
                disp_dict.update(layer_wb_dict_ade)

        total_loss_b = total_loss_b / self.num_decoder_layers
        total_loss_reg = total_loss_reg / self.num_decoder_layers
        total_loss_cls = total_loss_cls / self.num_decoder_layers


        return total_loss_b, total_loss_reg, total_loss_cls, wb_dict, disp_dict

    def get_loss(self, batch_dict, in_disp_dict, in_wb_dict, wb_pre_tag=''):
        # init
        flag_ctc_s1, flag_ctc_s2, denoiser_dict = self.get_ctc_data(batch_dict)

        denoiser_t = denoiser_dict['denoiser_t']                        # [B]

        # compute losses
        loss_decoder_b, loss_decoder_reg, loss_decoder_cls, wb_dict, disp_dict = self.get_decoder_loss(wb_pre_tag=wb_pre_tag)
        loss_dense_prediction, wb_dict, disp_dict = get_dense_future_prediction_loss(self.forward_ret_dict, wb_pre_tag, wb_dict, disp_dict)

        total_loss_reg = loss_decoder_reg + loss_dense_prediction
        total_loss_cls = loss_decoder_cls
        total_loss = total_loss_reg + total_loss_cls

        # update to the console and wandb
        if flag_ctc_s1:
            _entry_prefix = 'ctc_1_'
        elif flag_ctc_s2:
            _entry_prefix = 'ctc_2_'
        else:
            _entry_prefix = ''

        wb_dict[f'{wb_pre_tag}loss'] = total_loss.item()
        disp_dict[f'{wb_pre_tag}loss'] = total_loss.item()

        disp_dict = {f'{_entry_prefix}{k}': v for k, v in disp_dict.items()}
        wb_dict = {f'{_entry_prefix}{k}': v for k, v in wb_dict.items()}

        in_disp_dict.update(disp_dict)
        in_wb_dict.update(wb_dict)

        # record the loss for each denoising level
        if self.ctc_loss and flag_ctc_s1:
            loss_buffer_ = self.loss_buffer_ctc_1
        elif self.ctc_loss and flag_ctc_s2:
            loss_buffer_ = self.loss_buffer_ctc_2
        else:
            loss_buffer_ = self.loss_buffer

        flag_reset = loss_buffer_.record_loss(denoiser_t, loss_decoder_b.detach(), epoch_id=batch_dict['cur_epoch'])
        if flag_reset:
            dict_loss_per_level = loss_buffer_.get_average_loss()
            in_wb_dict.update({
                f'{_entry_prefix}denoiser_loss_per_level': dict_loss_per_level
            })

        return total_loss_reg, total_loss_cls

    def get_ctc_data(self, batch_dict):
        flag_ctc_s1 = self.ctc_loss and 'denoiser_dict_ctc_1' in batch_dict and 'denoiser_dict_ctc_2' not in batch_dict
        flag_ctc_s2 = self.ctc_loss and 'denoiser_dict_ctc_1' in batch_dict and 'denoiser_dict_ctc_2' in batch_dict
        if self.ctc_loss and self.training:
            if flag_ctc_s1:
                denoiser_dict = batch_dict['denoiser_dict_ctc_1']
            elif flag_ctc_s2:
                denoiser_dict = batch_dict['denoiser_dict_ctc_2']
            else:
                raise NotImplementedError("CTC loss input dict not found.")
        else:
            denoiser_dict = batch_dict['denoiser_dict']
        return flag_ctc_s1, flag_ctc_s2, denoiser_dict

    def forward(self, batch_dict):
        """Init"""
        flag_ctc_s1, flag_ctc_s2, denoiser_dict = self.get_ctc_data(batch_dict)

        """Process the denoising token and time embedding"""
        denoiser_x, denoiser_t = denoiser_dict['denoiser_x'], denoiser_dict['denoiser_t']

        # denoising timesteps embedding
        K = denoiser_x.shape[1]
        t_emb = self.t_embedder(denoiser_t)                                 # [B, D]
        t_emb = t_emb.unsqueeze(1).expand(-1, K, -1)                        # [B, K, D]

        # noisy vector embedding
        x_emb = rearrange(denoiser_x, 'b k l d -> b k (l d)')               # [B, K, T * 2] <- [B, K, T, 2]
        x_emb = self.x_embedder(x_emb)                                      # [B, K, D]     <- [B, K, T * 2]

        """Process the encoder output (context tokens)"""
        encoder_output = batch_dict['encoder_output']
        obj_feature, obj_mask, obj_pos = encoder_output['obj_feature'], encoder_output['obj_mask'], encoder_output['obj_pos']
        map_feature, map_mask, map_pos = encoder_output['map_feature'], encoder_output['map_mask'], encoder_output['map_pos']
        center_objects_feature = encoder_output['center_objects_feature']
        num_center_objects, num_objects, _ = obj_feature.shape
        num_polylines = map_feature.shape[1]

        # input projection - center objects
        center_objects_feature = self.in_proj_center_obj(center_objects_feature)

        # input projection - other objects
        obj_feature_valid = self.in_proj_obj(obj_feature[obj_mask])
        obj_feature = obj_feature.new_zeros(num_center_objects, num_objects, obj_feature_valid.shape[-1])
        obj_feature[obj_mask] = obj_feature_valid

        # input projection - map tokens
        map_feature_valid = self.in_proj_map(map_feature[map_mask])
        map_feature = map_feature.new_zeros(num_center_objects, num_polylines, map_feature_valid.shape[-1])
        map_feature[map_mask] = map_feature_valid

        # dense future prediction
        obj_feature, _ = apply_dense_future_prediction(
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos, forward_ret_dict=self.forward_ret_dict,
            obj_pos_encoding_layer=self.obj_pos_encoding_layer, dense_future_head=self.dense_future_head, 
            future_traj_mlps=self.future_traj_mlps, traj_fusion_mlps=self.traj_fusion_mlps, 
            num_future_frames=self.num_future_frames)

        """Go through cross-attention layers"""
        # embedding fusion
        center_objects_feature = center_objects_feature.unsqueeze(1).expand(-1, K, -1)  # [B, K, D]
        center_objects_feature = self.feat_fusion_mlp(torch.cat((x_emb, t_emb, center_objects_feature), dim=-1))  # [B, K, D]
        center_objects_feature = self.feat_tf_encoder(center_objects_feature)                                     # [B, K, D]

        input_dict = batch_dict['input_dict']
        if self.training:
            # add center gt trajs to the forward ret dict for get_center_gt_idx function
            self.forward_ret_dict['center_gt_trajs'] = input_dict['center_gt_trajs']
            self.forward_ret_dict['center_gt_trajs_mask'] = input_dict['center_gt_trajs_mask']
            self.forward_ret_dict['center_gt_final_valid_idx'] = input_dict['center_gt_final_valid_idx']
        pred_list = self.apply_transformer_decoder(
            center_objects_feature=center_objects_feature,
            center_objects_type=input_dict['center_objects_type'],
            obj_feature=obj_feature, obj_mask=obj_mask, obj_pos=obj_pos,
            map_feature=map_feature, map_mask=map_mask, map_pos=map_pos)

        """Update cache dict"""
        self.forward_ret_dict['pred_list'] = pred_list

        denoised_x_metric = pred_list[-1][1][..., :2]                                       # [B, K, T, 2]
        denoised_x = shift_data_to_normalize(denoised_x_metric, None, self.data_rescale)    # [B, K, T, 2]
        denoised_cls = pred_list[-1][0]  # [B, K]

        def _update_output_dict(denoised_x, denoised_cls, key='denoiser_output'):
            batch_dict[key] = {'denoised_x': denoised_x, 'denoised_cls': denoised_cls}
            return batch_dict

        if self.ctc_loss and self.training:
            if flag_ctc_s1:
                batch_dict = _update_output_dict(denoised_x, denoised_cls, key='denoiser_output_ctc_1')
            elif flag_ctc_s2:
                batch_dict = _update_output_dict(denoised_x, denoised_cls, key='denoiser_output_ctc_2')
            else:
                raise NotImplementedError("CTC loss input dict not found.")
        else:
            batch_dict = _update_output_dict(denoised_x, denoised_cls, key='denoiser_output')

        if not self.training:
            pred_scores, pred_trajs, selected_idxs = generate_final_prediction(pred_list, self.num_motion_modes)
            batch_dict['pred_scores'] = pred_scores
            batch_dict['pred_trajs'] = pred_trajs
            batch_dict['selected_idxs'] = selected_idxs
        else:
            self.forward_ret_dict['obj_trajs_future_state'] = input_dict['obj_trajs_future_state']
            self.forward_ret_dict['obj_trajs_future_mask'] = input_dict['obj_trajs_future_mask']
            self.forward_ret_dict['center_objects_type'] = input_dict['center_objects_type']

        return batch_dict

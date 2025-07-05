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


from typing import Optional
from copy import deepcopy

import torch
from torch import nn, Tensor
from .transformer_encoder_layer import _get_activation_fn

from trajflow.models.layers.common_layers import LlamaRMSNorm


def modulate(x, shift, scale):
    if len(x.shape) == 3 and len(shift.shape) == 2:
        # [B, K, D] + [B, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    elif len(x.shape) == len(shift.shape) == 3:
        # [B, K, D] + [B, K, D]
        return x * (1 + scale) + shift


class DMTDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 use_concat_pe_ca=True,
                 normalization_type="layer_norm", bias=True, qk_norm=False, adaLN=False):
        super().__init__()

        # Configurations
        self.nhead = nhead
        self.dropout_val = dropout
        self.normalize_before = normalize_before
        self.use_concat_pe_ca = use_concat_pe_ca
        self.normalization_type = normalization_type
        self.qk_norm = qk_norm
        self.adaLN = adaLN
        
        if normalize_before:
            raise NotImplementedError("normalize_before is not implemented yet.")
        
        if normalization_type == 'layer_norm':
            norm_layer = nn.LayerNorm(d_model, bias=bias)
        elif normalization_type == 'rms_norm':
            norm_layer = LlamaRMSNorm(d_model)

        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_qpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_kpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.sa_o_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_norm:
            self.sa_q_norm = deepcopy(norm_layer)
            self.sa_k_norm = deepcopy(norm_layer)

        self.norm1 = deepcopy(norm_layer)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_qpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_kpos_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model, bias=bias)
        self.ca_o_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_norm:
            d_model_ca = d_model * 2 if use_concat_pe_ca else d_model
            if normalization_type == 'layer_norm':
                qk_norm_layer = nn.LayerNorm(d_model_ca, bias=bias)
            elif normalization_type == 'rms_norm':
                qk_norm_layer = LlamaRMSNorm(d_model_ca)
            self.ca_q_norm = deepcopy(qk_norm_layer)
            self.ca_k_norm = deepcopy(qk_norm_layer)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm2 = deepcopy(norm_layer)
        self.norm3 = deepcopy(norm_layer)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Conditioning modulation layer
        # apply scale and shift modulation after self-attn, cross-attn and feedforward layers
        if self.adaLN:
            self.ada_ln = nn.Sequential(
                nn.SiLU(),
                nn.Linear(d_model, 6 * d_model, bias=True)
            )

            # zero initialization
            nn.init.constant_(self.ada_ln[1].weight, 0)
            nn.init.constant_(self.ada_ln[1].bias, 0)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, 
                query, 
                context,
                query_valid_mask=None,
                context_valid_mask=None,
                query_sa_pos_embeddings=None,                   # query self-attention PE embeddings
                query_ca_pos_embeddings=None,                   # query cross-attention coordinate-based PE embeddings
                context_ca_pos_embeddings=None,                 # context cross-attention coordinate-based PE embeddings
                adaln_emb=None,                                 # timestep embeddings
                is_causal=False,
                is_first=False,
                context_indexing=None,
                ):
        """
        @param query:               [B, N, D], query embeddings
        @param context:             [B, M, D], context embeddings
        @param query_valid_mask:    [B, N], query valid mask, True for valid, False for invalid
        @param context_valid_mask:  [B, M] or [B, N, M], context valid mask, True for valid, False for invalid
        @param query_sa_pos_embeddings: [B, N, D], query self-attention positional embeddings
        @param query_ca_pos_embeddings: [B, N, D], query cross-attention positional embeddings
        @param context_ca_pos_embeddings: [B, M, D], context cross-attention positional embeddings
        @param adaln_emb:           [B, D], timestep embeddings
        @param is_causal:           bool, whether it is the causal decoder 
        @param is_first:            bool, whether it is the first decoder layer
        @param context_indexing:    [B, N, Z], context indexing for the cross-attention layer
        """

        # Init
        if query_valid_mask is not None:
            assert query_valid_mask.dtype == torch.bool
            raise NotImplementedError("Masked self-attention is not implemented yet.")

        if context_valid_mask is not None:
            assert context_valid_mask.dtype == torch.bool
        assert context_valid_mask is not None, "Context valid mask is required."

        B, N, D = query.shape
        M = context.shape[1]

        # modulation
        if self.adaLN:
            assert adaln_emb is not None
            shift_sa, scale_sa, shift_ca, scale_ca, shift_ff, scale_ff = self.ada_ln(adaln_emb).chunk(6, dim=-1)  # [B, *, D] * 6

        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: [B, N, D]
        sa_q_content = self.sa_qcontent_proj(query)                    # query is the input of the first decoder layer. zero by default.
        sa_q_pos = self.sa_qpos_proj(query_sa_pos_embeddings)
        sa_k_content = self.sa_kcontent_proj(query)
        sa_k_pos = self.sa_kpos_proj(query_sa_pos_embeddings)
        sa_v = self.sa_v_proj(query)

        # additive positional embedding
        sa_q = sa_q_content + sa_q_pos
        sa_k = sa_k_content + sa_k_pos

        # QK norm if needed
        if self.qk_norm:
            sa_q = self.sa_q_norm(sa_q)
            sa_k = self.sa_k_norm(sa_k)

        # reshape for multi-head attention
        sa_q = sa_q.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead], sequence length is the second-from-last dimension
        sa_k = sa_k.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead]
        sa_v = sa_v.view(B, N, self.nhead, D // self.nhead).transpose(1, 2)  # [B, nhead, N, D//nhead]
        
        # scaled dot-product attention
        tgt2 = torch.nn.functional.scaled_dot_product_attention(sa_q, sa_k, sa_v,
                                                                is_causal=is_causal,
                                                                dropout_p=self.dropout_val if self.training else 0.0)  # [B, nhead, N, D//nhead]
        tgt2 = tgt2.transpose(1, 2).contiguous().view(B, N, D)  # [B, N, D]
        tgt2 = self.sa_o_proj(tgt2)
        # ========== End of Self-Attention =============

        query = query + self.dropout1(tgt2)                      # residual connection
        query = self.norm1(query)                                # [B, N, D]

        # modulation
        if self.adaLN:
            query = modulate(query, shift_sa, scale_sa)


        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: [B, M/N, D]

        ## query projection
        ca_q_content = self.ca_qcontent_proj(query)                     # [B, N, D]
        ca_q_pos = self.ca_qpos_sine_proj(query_ca_pos_embeddings)      # [B, N, D]

        ## key and value projection by selecting valid context tokens only
        if len(context_valid_mask.shape) == 3:
            assert context_valid_mask.shape[1] == N

            context_valid_mask_ = context_valid_mask.any(dim=1)             # [B, M]
            context_invalid_mask_ = torch.logical_not(context_valid_mask_)  # [B, M]
            context_invalid_mask_ = context_invalid_mask_.unsqueeze(-1)     # [B, M, 1]

            ## key projection
            ca_k_content = self.ca_kcontent_proj(context)                               # [B, M, D], unmasked context
            ca_k_content = torch.masked_fill(ca_k_content, context_invalid_mask_, 0.0)  # [B, M, D], masked context

            ## value projection
            ca_v = self.ca_v_proj(context)                                # [B, M, D], unmasked context
            ca_v = torch.masked_fill(ca_v, context_invalid_mask_, 0.0)    # [B, M, D], masked context
            
            ## positional embedding projection
            ca_k_pos = self.ca_kpos_proj(context_ca_pos_embeddings)               # [B, M, D], unmasked context
            ca_k_pos = torch.masked_fill(ca_k_pos, context_invalid_mask_, 0.0)    # [B, M, D], masked context
        elif len(context_valid_mask.shape) == 2:
            valid_context = context[context_valid_mask]  # [B, M, D] -> [M1 + M2 + ... + Mn, D]

            ## key projection
            ca_k_content_valid = self.ca_kcontent_proj(valid_context)
            ca_k_content = context.new_zeros(B, M, D)
            ca_k_content[context_valid_mask] = ca_k_content_valid

            ## value projection
            ca_v_valid = self.ca_v_proj(valid_context)
            ca_v = context.new_zeros(B, M, D)
            ca_v[context_valid_mask] = ca_v_valid

            ## positional embedding projection
            ca_valid_pos = context_ca_pos_embeddings[context_valid_mask]
            ca_k_pos_valid = self.ca_kpos_proj(ca_valid_pos)
            ca_k_pos = context_ca_pos_embeddings.new_zeros(B, M, D)
            ca_k_pos[context_valid_mask] = ca_k_pos_valid
        else:
            raise ValueError("Invalid context_valid_mask shape.")

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first:
            # raise NotImplementedError("This module is not used at all!")
            ### we may want to remove this block later ###
            ca_q_pos_from_sa = self.ca_qpos_proj(query_sa_pos_embeddings)  # note this is using the self-attention positional embeddings
            ca_q = ca_q_content + ca_q_pos_from_sa  # [B, N, D]
            ca_k = ca_k_content + ca_k_pos          # [B, M, D]
            ### we may want to remove this block later ###
        else:
            ca_q = ca_q_content                     # [B, N, D]
            ca_k = ca_k_content                     # [B, M, D]

        if self.use_concat_pe_ca:
            # concatenating positional embeddings
            ca_q = torch.cat([ca_q, ca_q_pos], dim=-1)  # [B, N, D*2]
            ca_k = torch.cat([ca_k, ca_k_pos], dim=-1)  # [B, M, D*2]
            D_cat = D * 2
        else:
            # adding positional embeddings
            ca_q = ca_q + ca_q_pos  # [B, N, D]
            ca_k = ca_k + ca_k_pos  # [B, M, D]
            D_cat = D

        # QK norm if needed
        if self.qk_norm:
            ca_q = self.ca_q_norm(ca_q)
            ca_k = self.ca_k_norm(ca_k)

        ### deprecated ###
        # if len(context_valid_mask.shape) == 3:
            # # stack context tokens for B * K batch to reduce computation
            # Z = context_indexing.shape[2]
            # _zero_pad = torch.zeros([B, 1, D_cat], dtype=ca_k.dtype, device=ca_k.device)
            # ca_k_pad = torch.cat([_zero_pad, ca_k], dim=1)      # [B, M+1, D]
            # _zero_pad = torch.zeros([B, 1, D], dtype=ca_k.dtype, device=ca_k.device)
            # ca_v_pad = torch.cat([_zero_pad, ca_v], dim=1)      # [B, M+1, C]

            # ca_k_stacked = torch.gather(input=ca_k_pad[:, None, :, :].expand(-1, N, -1, -1), dim=2, index=(1 + context_indexing).long().unsqueeze(-1).expand(-1, -1, -1, D_cat))    # [B, N, Z, D]
            # ca_v_stacked = torch.gather(input=ca_v_pad[:, None, :, :].expand(-1, N, -1, -1), dim=2, index=(1 + context_indexing).long().unsqueeze(-1).expand(-1, -1, -1, D))        # [B, N, Z, C]
            # ca_k_stacked_valid_mask = context_indexing != -1  # [B, N, Z]

            # # sanity check #
            # # assert ca_k_stacked[0, 1, 2].equal(ca_k_pad[0, (1 + context_indexing).long()[0, 1, 2]])
            # # assert ca_k_stacked[-1, -2, -3].equal(ca_k_pad[-1, (1 + context_indexing).long()[-1, -2, -3]])
            # # assert (ca_k_stacked[torch.logical_not(ca_k_stacked_valid_mask)] == 0).all()
            # # assert (ca_v_stacked[torch.logical_not(ca_k_stacked_valid_mask)] == 0).all()
            # # sanity check #

            # # in the stacked mode, we let the query token have unit length and attend to Z context tokens independently
            # ca_q = ca_q.view(B, N, self.nhead, 1, D_cat//self.nhead)                                    # [B, N, nhead, 1, D//nhead], length is 1
            # ca_k = ca_k_stacked.view(B, N, Z, self.nhead, D_cat//self.nhead).permute(0, 1, 3, 2, 4)     # [B, N, nhead, Z, D//nhead], length is Z
            # ca_v = ca_v_stacked.view(B, N, Z, self.nhead, D//self.nhead).permute(0, 1, 3, 2, 4)         # [B, N, nhead, Z, D//nhead], length is Z

            # attn_mask_bool = ca_k_stacked_valid_mask[:, :, None, None, :].expand(-1, -1, self.nhead, -1, -1)  # [B, N, nhead, 1, Z]

            # # scaled dot-product attention
            # tgt2 = torch.nn.functional.scaled_dot_product_attention(ca_q, ca_k, ca_v, 
            #                                                         attn_mask=attn_mask_bool,
            #                                                         dropout_p=self.dropout_val if self.training else 0.0)  # [B, N, nhead, 1, D//nhead]
            # tgt2 = tgt2.contiguous().view(B, N, D)
        ### deprecated ###
            
        # reshape for multi-head attention
        ca_q = ca_q.view(B, N, self.nhead, D_cat//self.nhead).transpose(1, 2)   # [B, nhead, N, D//nhead], sequence length is the second-from-last dimension
        ca_k = ca_k.view(B, M, self.nhead, D_cat//self.nhead).transpose(1, 2)   # [B, nhead, M, D//nhead]
        ca_v = ca_v.view(B, M, self.nhead, D//self.nhead).transpose(1, 2)       # [B, nhead, M, D//nhead]
        if len(context_valid_mask.shape) == 3:
            attn_mask_bool = context_valid_mask[:, None, :, :].expand(B, self.nhead, N, M)      # [B, nhead, N, M]
        elif len(context_valid_mask.shape) == 2:
            attn_mask_bool = context_valid_mask[:, None, None, :].expand(B, self.nhead, N, M)   # [B, nhead, N, M]
        else:
            raise ValueError("Invalid context_valid_mask shape.")

        # scaled dot-product attention
        tgt2 = torch.nn.functional.scaled_dot_product_attention(ca_q, ca_k, ca_v, 
                                                                attn_mask=attn_mask_bool,
                                                                dropout_p=self.dropout_val if self.training else 0.0)  # [B, nhead, N, D//nhead]
        tgt2 = tgt2.transpose(1, 2).contiguous().view(B, N, D)
        tgt2 = self.ca_o_proj(tgt2)
        # ========== End of Cross-Attention =============

        query = query + self.dropout2(tgt2)
        query = self.norm2(query)

        # modulation
        if self.adaLN:
            query = modulate(query, shift_ca, scale_ca)


        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(tgt2)
        query = self.norm3(query)

        # modulation
        if self.adaLN:
            query = modulate(query, shift_ff, scale_ff)

        return query


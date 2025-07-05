# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch
import torch.nn as nn

from trajflow.models import MTREncoder, DenoisingDecoder
from trajflow.utils.common_utils import register_module_to_params_dict


class DenoisingMotionTransformer(nn.Module):
    def __init__(self, config, logger):
        super().__init__()

        # init
        self.config = config

        self.model_cfg = config.MODEL_DMT
        self.model_cfg.CONTEXT_ENCODER.DEVICE = config.DEVICE
        self.model_cfg.DMT.DEVICE = config.DEVICE

        self.model_cfg.DMT.CONTEXT_D_MODEL = self.model_cfg.CONTEXT_ENCODER.D_MODEL

        self.ctc_loss = self.model_cfg.DENOISING.CTC_LOSS

        self.logger = logger

        # Encoder network (reusing the MTR encoder)
        self.context_encoder = MTREncoder(self.model_cfg.CONTEXT_ENCODER)

        # Denoising decoder
        self.denoising_decoder = DenoisingDecoder(model_cfg=self.model_cfg.DMT, denoising_cfg=self.model_cfg.DENOISING,
                                                  logger=logger, save_dirs=self.config.SAVE_DIR, data_rescale=config.DATA_CONFIG.DATA_RESCALE)

        self.params_dict = {}
        self.register_module_to_params_dict = lambda module, name: register_module_to_params_dict(self.params_dict, module, name)
        self.count_model_params()
         
    def count_model_params(self):
        """
        Count the number of trainable parameters in the model.
        """
        self.logger.info("===== Overall DMT model parameters breakdown =====")
        params_total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.register_module_to_params_dict(self.context_encoder, 'MTR_encoder')
        self.register_module_to_params_dict(self.denoising_decoder, 'denoising_decoder')

        params_other = params_total - sum(self.params_dict.values())
        self.params_dict = {'total': params_total, **self.params_dict}
        self.params_dict = {**self.params_dict, 'other': params_other}
        for nm, p in self.params_dict.items():
            self.logger.info("#params for {:40}: {:,}".format(nm, p))

        self.logger.info("===== Overall DMT model parameters breakdown =====")
 
    def forward(self, batch_dict, disp_dict=None, wb_dict=None):
        """
        Forward pass of the model.
        """

        if self.training:
            # get ctc loss flags
            if self.ctc_loss:
                flag_ctc_s1 = 'denoiser_dict_ctc_1' in batch_dict and 'denoiser_dict_ctc_2' not in batch_dict
                flag_ctc_s2 = 'denoiser_dict_ctc_1' in batch_dict and 'denoiser_dict_ctc_2' in batch_dict
                assert flag_ctc_s1 or flag_ctc_s2, 'CTC loss is not properly set'
        
            """context encoder"""
            if self.ctc_loss and flag_ctc_s2:
                # use the cached encoder output for CTC loss
                assert 'encoder_output' in batch_dict, 'encoder_output is not found in batch_dict'
            else:
                batch_dict = self.context_encoder(batch_dict)       # batch dict is updated with new outputs

            """denoising decoder"""
            batch_dict = self.denoising_decoder(batch_dict)         # batch dict is updated with new outputs

            """compute loss"""
            loss_denoiser_reg, loss_denoiser_cls = self.denoising_decoder.get_loss(batch_dict, disp_dict, wb_dict)

            return loss_denoiser_reg, loss_denoiser_cls, batch_dict

        else:
            flag_run_encoder_net = 'encoder_output' not in batch_dict

            """context encoder"""
            if flag_run_encoder_net:
                batch_dict = self.context_encoder(batch_dict)           # batch dict is updated with new outputs

            """denoising decoder"""
            # always run this module
            batch_dict = self.denoising_decoder(batch_dict)             # batch dict is updated with new outputs

            return batch_dict

    def load_params(self, ckpt_path, to_cpu=True, ckpt_state=None, optimizer=None, ema_model_kw=None, ema_helper=None):
        """
        Helper to load model parameters, optimizer states, and EMA model states from a checkpoint.
        """
        # init and load checkpoint into memory
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError
        
        if ckpt_state is not None:
            self.logger.info('==> Loading parameters from in-memory checkpoint dict...')
            checkpoint = ckpt_state
        else:
            self.logger.info('==> Loading parameters from checkpoint %s to %s' % (ckpt_path, 'CPU' if to_cpu else 'GPU'))
            loc_type = torch.device('cpu') if to_cpu else None
            checkpoint = torch.load(ckpt_path, map_location=loc_type)

        epoch = checkpoint.get('epoch', -1)
        it = checkpoint.get('it', 0.0)
        version = checkpoint.get("version", None)

        if version is not None:
            self.logger.info('==> Checkpoint trained from version: %s' % version)

        # load EMA model if needed
        if ema_model_kw is not None:
            assert ema_model_kw in checkpoint, f'key {ema_model_kw} not found in checkpoint'
            loaded_model_state = checkpoint[ema_model_kw]
            self.logger.info(f'==> Loading EMA model with key {ema_model_kw} from checkpoint')
        else:
            loaded_model_state = checkpoint['model_state']

        # check the keys in the checkpoint
        self.logger.info(f'The number of in-memory ckpt keys: {len(loaded_model_state)}')
        cur_model_state = self.state_dict()
        loaded_model_state_filtered = {}
        for key, val in loaded_model_state.items():
            if key in cur_model_state and loaded_model_state[key].shape == cur_model_state[key].shape:
                loaded_model_state_filtered[key] = val
            else:
                if key not in cur_model_state:
                    self.logger.info(f'Ignore key in disk (not found in model): {key}, shape={val.shape}')
                else:
                    self.logger.info(f'Ignore key in disk (shape does not match): {key}, load_shape={val.shape}, model_shape={cur_model_state[key].shape}')

        # load the filtered checkpoint
        missing_keys, unexpected_keys = self.load_state_dict(loaded_model_state_filtered, strict=True)

        self.logger.info(f'Missing keys: {missing_keys}')
        self.logger.info(f'The number of missing keys: {len(missing_keys)}')
        self.logger.info(f'The number of unexpected keys: {len(unexpected_keys)}')
        self.logger.info('==> Done loading model ckpt (total keys %d)' % (len(cur_model_state)))

        # laod optimizer if needed
        if optimizer is not None:
            self.logger.info('==> Loading optimizer parameters from checkpoint %s to %s' % (ckpt_path, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.logger.info('==> Done loading optimizer state')

        # load EMA helper weights if needed
        if ema_helper is not None:
            for ema_wrapper in ema_helper:
                beta = ema_wrapper.beta
                self.logger.info('==> Loading EMA model with beta = %.4f from checkpoint %s to %s'% (beta, ckpt_path, 'CPU' if to_cpu else 'GPU'))
                ema_wrapper.ema_model.load_state_dict(checkpoint['model_ema_beta_{:.4f}'.format(beta)], strict=True)
            self.logger.info('==> Done loading EMA model')

        return it, epoch

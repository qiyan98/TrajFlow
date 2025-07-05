# Copyright (c) 2025-present, Qi Yan.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn
import torch.distributed as dist
from collections import namedtuple

from trajflow.utils.denoising_data_rescale import shift_data_to_denormalize

# Constants
ModelPrediction = namedtuple('ModelPrediction', ['pred_vel', 'pred_data', 'x_cls'])

# Helper functions
def exists(x): return x is not None

def default(val, d):
    if exists(val): return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)): return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class FlowMatcher(nn.Module):
    def __init__(
        self,
        model,
        *,
        logger=None,
        data_rescale='linear',
        ckpt_dir=None,
        model_cfg=None,
        sampling_timesteps=None,
        objective='pred_data',
        t_schedule='uniform',
    ):
        super().__init__()
        
        # Configuration
        self.logger = logger
        self.data_rescale = data_rescale
        self.ckpt_dir = ckpt_dir
        self.model_cfg = model_cfg
        self.model = model
        
        # Model parameters
        self.traj_future_frames = model_cfg.DMT.NUM_FUTURE_FRAMES
        self.traj_waypoint_dim = 2
        self.num_query = model_cfg.DMT.NUM_QUERY
        self.tied_noise = model_cfg.DENOISING.TIED_NOISE
        self.ctc_loss = model_cfg.DENOISING.CTC_LOSS
        
        # Training parameters
        self.objective = objective
        self.t_schedule = t_schedule
        self.sampling_timesteps = sampling_timesteps
        
        # Validation
        assert objective in {'pred_data', 'pred_vel'}, 'objective must be either pred_data or pred_vel'
        assert t_schedule in {'uniform', 'log_normal'}, 't_schedule must be either uniform or log_normal'

    @property
    def device(self):
        return self.model_cfg.DEVICE
    
    def predict_vel_from_data(self, x1, xt, t):
        """Predict velocity field from predicted data."""
        t = pad_t_like_x(t, x1)
        return (x1 - xt) / (1 - t)
    
    def predict_data_from_vel(self, v, xt, t):
        """Predict data from predicted velocity field."""
        t = pad_t_like_x(t, xt)
        return xt + v * (1 - t)

    def fwd_sample_t(self, x0, x1, t):
        """
        Sample the latent space at time t.
        """
        t = pad_t_like_x(t, x0)
        xt = t * x1 + (1 - t) * x0      # simple linear interpolation
        ut = x1 - x0                    # xt derivative w.r.t. t
        return xt, ut

    def _get_time_steps(self, batch_size, flag_t_is_0=False):
        """Generate time steps for training."""
        if flag_t_is_0:
            return torch.zeros((batch_size,), device=self.device)
        
        if self.t_schedule == 'uniform':
            return torch.rand((batch_size,), device=self.device)
        elif self.t_schedule == 'log_normal':
            t_normal = torch.randn((batch_size,), device=self.device)
            return torch.sigmoid(t_normal)

    def _get_noise(self, x_data, specify_noise=None):
        """Generate noise for training."""
        x_data_k = x_data.unsqueeze(1).expand(-1, self.num_query, -1, -1)
        
        if specify_noise is not None:
            return specify_noise
        
        if self.tied_noise:
            noise = torch.randn_like(x_data_k[:, 0:1])
            return noise.expand(-1, self.num_query, -1, -1)
        else:
            return torch.randn_like(x_data_k)

    def get_loss_input(self, batch, flag_t_is_0=False, specify_noise=None, specify_pseudo_data=None):
        """Prepare input for flow matching model training."""
        batch_size = sum(batch['batch_sample_count'])
        
        # Generate time steps
        t = self._get_time_steps(batch_size, flag_t_is_0)
        assert t.min() >= 0 and t.max() <= 1
        
        # Get data and noise
        x_data = batch['denoiser_dict']['gt_traj_normalized'].to(self.device)
        noise = self._get_noise(x_data, specify_noise)
        x_data_k = x_data.unsqueeze(1).expand(-1, self.num_query, -1, -1)

        # Sample latent space
        if specify_pseudo_data is not None:
            x_t, u_t = self.fwd_sample_t(x0=noise, x1=specify_pseudo_data, t=t)
        else:
            x_t, u_t = self.fwd_sample_t(x0=noise, x1=x_data_k, t=t)
        
        # Set target based on objective
        target = x_data_k if self.objective == 'pred_data' else u_t
        l_weight = torch.ones_like(t)
        
        return t, x_t, u_t, target, noise, l_weight
    
    def _update_denoiser_dict(self, batch_dict, t, x_t, u_t, target, l_weight, key='denoiser_dict'):
        """Update denoiser dictionary with training data."""
        if key != 'denoiser_dict':
            batch_dict[key] = {
                'gt_traj_metric': batch_dict['denoiser_dict']['gt_traj_metric'],
                'gt_traj_normalized': batch_dict['denoiser_dict']['gt_traj_normalized'],
                'gt_traj_mask': batch_dict['denoiser_dict']['gt_traj_mask'],
            }
        
        batch_dict[key].update({
            'denoiser_t': t,
            'denoiser_x': x_t,
            'denoiser_gt_vel': u_t,
            'denoiser_target': target,
            'denoiser_l_weight': l_weight
        })

    def model_predictions(self, x, t, batch_dict, last_step):
        """Get model predictions for given input."""
        batch_dict['denoiser_dict'].update({
            'denoiser_t': t,
            'denoiser_x': x,
            'last_step': last_step,
        })
        
        batch_dict = self.model(batch_dict)
        denoiser_output = batch_dict['denoiser_output']
        denoised_x = denoiser_output['denoised_x']
        denoised_cls = denoiser_output['denoised_cls']
        
        assert denoised_x.shape == x.shape
        
        # Generate predictions based on objective
        if self.objective == 'pred_data':
            pred_data = denoised_x
            pred_vel = self.predict_vel_from_data(x1=pred_data, xt=x, t=t)
        elif self.objective == 'pred_vel':
            pred_vel = denoised_x
            pred_data = self.predict_data_from_vel(v=pred_vel, xt=x, t=t)
        
        return ModelPrediction(pred_vel, pred_data, denoised_cls)
    
    @torch.inference_mode()
    def bwd_sample_t(self, x, t, dt, batch_dict, last_step):
        """Backward sampling step."""
        B = x.shape[0]
        batched_t = torch.full((B,), t, device=self.device, dtype=torch.float)
        model_preds = self.model_predictions(x, batched_t, batch_dict, last_step)
        x_next = x + model_preds.pred_vel * dt
        return x_next, model_preds
    
    @torch.inference_mode()
    def sample(self, batch_dict, return_all_timesteps=False):
        """Draw samples from the flow matching model."""
        batch_size = sum(batch_dict['batch_sample_count'])
        shape = (batch_size, self.num_query, self.traj_future_frames, self.traj_waypoint_dim)
        
        # Initialize noise
        B, K, T, D = shape
        if self.tied_noise:
            init_noise = torch.randn((B, 1, T, D), device=self.device).expand(-1, K, -1, -1)
        else:
            init_noise = torch.randn(shape, device=self.device)
        
        # Sampling loop
        traj = init_noise
        all_trajs = [traj]
        dt = 1.0 / self.sampling_timesteps
        
        for idx_t in range(self.sampling_timesteps):
            t = dt * idx_t
            last_step = (idx_t + 1) == self.sampling_timesteps
            traj, preds = self.bwd_sample_t(traj, t, dt, batch_dict, last_step)
            all_trajs.append(traj)
        
        ret = traj if not return_all_timesteps else torch.stack(all_trajs, dim=1)
        return ret, preds.x_cls

    def forward(self, batch_dict, disp_dict=None, wb_dict=None, flag_sample=False):
        """Main forward function for flow matching module."""
        if flag_sample:
            # Generate trajectories
            trajs, cls = self.sample(batch_dict)
            trajs = shift_data_to_denormalize(trajs, None, data_rescale=self.data_rescale)
            return trajs, cls, batch_dict
        
        # Training mode
        if self.ctc_loss:
            return self._forward_ctc_loss(batch_dict, disp_dict, wb_dict)
        else:
            return self._forward_standard_loss(batch_dict, disp_dict, wb_dict)
    
    def _forward_ctc_loss(self, batch_dict, disp_dict, wb_dict):
        """Forward pass with cross-time consistency loss."""
        assert self.objective == 'pred_data', 'cross-time consistency loss only supports pred_data objective'
        # First round: enforce time = 0 prediction
        t, x_t, u_t, target, noise, l_weight = self.get_loss_input(batch_dict, flag_t_is_0=True)
        self._update_denoiser_dict(batch_dict, t, x_t, u_t, target, l_weight, key='denoiser_dict_ctc_1')
        ctc_1_loss_reg, ctc_1_loss_cls, batch_dict = self.model(batch_dict, disp_dict, wb_dict)
        
        # Second round: enforce arbitrary time prediction
        random_val = torch.rand(1).item()
        if dist.is_initialized():  # Synchronize random value in DDP mode
            random_val_tensor = torch.tensor(random_val).to(self.device)
            dist.broadcast(random_val_tensor, src=0)
            random_val = random_val_tensor.item()
        flag_second_round = random_val < 0.5

        if flag_second_round:
            data_pred = batch_dict['denoiser_output_ctc_1']['denoised_x'].detach()
            t, x_t, u_t, target, noise, l_weight = self.get_loss_input(
                batch_dict, specify_noise=noise, specify_pseudo_data=data_pred)
            self._update_denoiser_dict(batch_dict, t, x_t, u_t, target, l_weight, key='denoiser_dict_ctc_2')
            ctc_2_loss_reg, ctc_2_loss_cls, batch_dict = self.model(batch_dict, disp_dict, wb_dict)
        else:
            ctc_2_loss_reg = torch.zeros_like(ctc_1_loss_reg)
            ctc_2_loss_cls = torch.zeros_like(ctc_1_loss_cls)
        return ctc_1_loss_reg + ctc_2_loss_reg, ctc_1_loss_cls + ctc_2_loss_cls, batch_dict
    
    def _forward_standard_loss(self, batch_dict, disp_dict, wb_dict):
        """Forward pass with standard loss."""
        t, x_t, u_t, target, noise, l_weight = self.get_loss_input(batch_dict)
        self._update_denoiser_dict(batch_dict, t, x_t, u_t, target, l_weight)
        return self.model(batch_dict, disp_dict, wb_dict)
    

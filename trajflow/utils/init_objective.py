from .denoising_data_rescale import shift_data_to_normalize, shift_data_to_denormalize


def prepare_denoiser_data(batch_dict, data_rescale, device):
    """
    Retrieve and prepare the relevant data for the denoising model.
    """
    bs = sum(batch_dict['batch_sample_count'])

    # normalize GT trajectory
    gt_traj_metric = batch_dict['input_dict']['center_gt_trajs'][..., :2].to(device)            # [B, T, 2]
    gt_traj_mask = batch_dict['input_dict']['center_gt_trajs_mask'].to(device)                  # [B, T]
    gt_traj_normalized = shift_data_to_normalize(gt_traj_metric, gt_traj_mask.bool(), data_rescale)                # [B, T, 2]


    """update data dict"""
    denoiser_dict = {
        'gt_traj_metric': gt_traj_metric,                                   # [B, T, 2]
        'gt_traj_normalized': gt_traj_normalized,                           # [B, T, 2]
        'gt_traj_mask': gt_traj_mask,                                       # [B, T]
    }
    batch_dict['denoiser_dict'] = denoiser_dict
        
    return batch_dict
    
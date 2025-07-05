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
import logging
import torch.distributed as dist
import random
import os
import pickle
import shutil

from torch.nn import functional as F


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    if points.shape[-1] == 2:
        rot_matrix = torch.stack((
            cosa,  sina,
            -sina, cosa
        ), dim=1).view(-1, 2, 2).float()
        points_rot = torch.matmul(points, rot_matrix)
    else:
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def merge_batch_by_padding_2nd_dim(tensor_list, return_pad_mask=False):
    assert len(tensor_list[0].shape) in [3, 4]
    only_3d_tensor = False
    if len(tensor_list[0].shape) == 3:
        tensor_list = [x.unsqueeze(dim=-1) for x in tensor_list]
        only_3d_tensor = True
    maxt_feat0 = max([x.shape[1] for x in tensor_list])

    _, _, num_feat1, num_feat2 = tensor_list[0].shape

    ret_tensor_list = []
    ret_mask_list = []
    for k in range(len(tensor_list)):
        cur_tensor = tensor_list[k]
        assert cur_tensor.shape[2] == num_feat1 and cur_tensor.shape[3] == num_feat2

        new_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0, num_feat1, num_feat2)
        new_tensor[:, :cur_tensor.shape[1], :, :] = cur_tensor
        ret_tensor_list.append(new_tensor)

        new_mask_tensor = cur_tensor.new_zeros(cur_tensor.shape[0], maxt_feat0)
        new_mask_tensor[:, :cur_tensor.shape[1]] = 1
        ret_mask_list.append(new_mask_tensor.bool())

    ret_tensor = torch.cat(ret_tensor_list, dim=0)  # (num_stacked_samples, num_feat0_maxt, num_feat1, num_feat2)
    ret_mask = torch.cat(ret_mask_list, dim=0)

    if only_3d_tensor:
        ret_tensor = ret_tensor.squeeze(dim=-1)

    if return_pad_mask:
        return ret_tensor, ret_mask
    return ret_tensor


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size


def get_batch_offsets(batch_idxs, bs):
    '''
    :param batch_idxs: (N), int
    :param bs: int
    :return: batch_offsets: (bs + 1)
    '''
    batch_offsets = torch.zeros(bs + 1).int().cuda()
    for i in range(bs):
        batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
    assert batch_offsets[-1] == batch_idxs.shape[0]
    return batch_offsets


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dist_pytorch(local_rank, backend='nccl'):
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()

    total_gpus = dist.get_world_size()
    return total_gpus, rank


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def apply_chunk_map_collection(map_pos, map_mask, base_region_offset, num_chunks):
    """
    Divide the map tokens into multiple chunks.
    @param map_pos:                     [B, M, 3], map position
    @param map_mask:                    [B, M], map mask
    @param base_region_offset:          [2], base region offset
    @param num_chunks:                  int, number of chunks
    """

    # Init
    B, M = map_pos.shape[:2]
    size_per_chunk = np.ceil(M / num_chunks).astype(int)
    padded_size = size_per_chunk * num_chunks
    if size_per_chunk * num_chunks != M:
        map_pos_chunk = F.pad(map_pos, pad=(0, 0, 0, padded_size - M), mode='constant', value=10000000.0)
        map_mask_chunk = F.pad(map_mask, pad=(0, padded_size - M), mode='constant', value=False)
    else:
        map_pos_chunk = map_pos.clone()
        map_mask_chunk = map_mask.clone()

    map_pos_chunk[~map_mask_chunk] = 10000000.0    # set the masked position to a large value

    base_points = torch.tensor(base_region_offset).type_as(map_pos_chunk)               # [2]
    base_dist = (map_pos_chunk[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)    # [B, M]
    base_topk_dist, base_map_idxs = base_dist.topk(k=padded_size, dim=-1, largest=False)          # [B, M]
    base_map_idxs[base_topk_dist > 10000000] = -1

    num_valid_tokens = (base_map_idxs != -1).sum(dim=-1)                                # [B]
    num_valid_tokens_per_chunk = num_valid_tokens // num_chunks                         # [B]

    # balance the valid tokens for each chunk
    chunk_map_idxs = []
    for i_chunk in range(num_chunks):
        this_chunk_map_idxs = []
        for i_batch in range(B):
            this_valid_token_start = i_chunk * num_valid_tokens_per_chunk[i_batch]
            this_valid_token_end = (i_chunk + 1) * num_valid_tokens_per_chunk[i_batch]
            this_base_map_idxs = base_map_idxs[i_batch][this_valid_token_start:this_valid_token_end]
            this_base_map_idxs = F.pad(this_base_map_idxs, pad=(0, size_per_chunk - len(this_base_map_idxs)), mode='constant', value=-1)  # [size_per_chunk]
            this_chunk_map_idxs.append(this_base_map_idxs)
        this_chunk_map_idxs = torch.stack(this_chunk_map_idxs, dim=0).unsqueeze(1).int()  # [B, 1, size_per_chunk]
        chunk_map_idxs.append(this_chunk_map_idxs)

    return chunk_map_idxs
   

def apply_dynamic_map_collection(map_pos, map_mask, pred_waypoints, base_region_offset, num_query, num_waypoint_polylines=128, num_base_polylines=256, base_map_idxs=None):
    map_pos = map_pos.clone()
    map_pos[~map_mask] = 10000000.0
    num_polylines = map_pos.shape[1]

    if base_map_idxs is None:
        base_points = torch.tensor(base_region_offset).type_as(map_pos)
        base_dist = (map_pos[:, :, 0:2] - base_points[None, None, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
        base_topk_dist, base_map_idxs = base_dist.topk(k=min(num_polylines, num_base_polylines), dim=-1, largest=False)  # (num_center_objects, topk)
        base_map_idxs[base_topk_dist > 10000000] = -1
        base_map_idxs = base_map_idxs[:, None, :].repeat(1, num_query, 1)  # (num_center_objects, num_query, num_base_polylines)
        if base_map_idxs.shape[-1] < num_base_polylines:
            base_map_idxs = F.pad(base_map_idxs, pad=(0, num_base_polylines - base_map_idxs.shape[-1]), mode='constant', value=-1)

    dynamic_dist = (pred_waypoints[:, :, None, :, 0:2] - map_pos[:, None, :, None, 0:2]).norm(dim=-1)  # (num_center_objects, num_query, num_polylines, num_timestamps)
    dynamic_dist = dynamic_dist.min(dim=-1)[0]  # (num_center_objects, num_query, num_polylines)

    dynamic_topk_dist, dynamic_map_idxs = dynamic_dist.topk(k=min(num_polylines, num_waypoint_polylines), dim=-1, largest=False)
    dynamic_map_idxs[dynamic_topk_dist > 10000000] = -1
    if dynamic_map_idxs.shape[-1] < num_waypoint_polylines:
        dynamic_map_idxs = F.pad(dynamic_map_idxs, pad=(0, num_waypoint_polylines - dynamic_map_idxs.shape[-1]), mode='constant', value=-1)

    collected_idxs = torch.cat((base_map_idxs, dynamic_map_idxs), dim=-1)  # (num_center_objects, num_query, num_collected_polylines)

    # remove duplicate indices
    sorted_idxs = collected_idxs.sort(dim=-1)[0]
    duplicate_mask_slice = (sorted_idxs[..., 1:] - sorted_idxs[..., :-1] != 0)  # (num_center_objects, num_query, num_collected_polylines - 1)
    duplicate_mask = torch.ones_like(collected_idxs).bool()
    duplicate_mask[..., 1:] = duplicate_mask_slice
    sorted_idxs[~duplicate_mask] = -1

    return sorted_idxs.int(), base_map_idxs


def count_trainable_params(module, verbose=False):
    if isinstance(module, torch.nn.Module):
        sum_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        sum_params = 0
    if verbose:
        print("number of trainable parameters: ", sum_params)
    return sum_params


def register_module_to_params_dict(params_dict, module, name):
    """
    Register the number of parameters in a module to the parameter count dictionary.
    """
    params_dict[name] = count_trainable_params(module)
    return params_dict


def log_gpu_memory_usage(custom_msg, logger):
    """
    Logs the current GPU memory usage using the provided logger.

    Args:
        logger (logging.Logger): The logger instance to use for logging the memory usage.
    """
    allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
    max_allocated_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # Convert to MB
    reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)  # Convert to MB
    max_reserved_memory = torch.cuda.max_memory_reserved() / (1024 ** 2)  # Convert to MB

    logger.info("GPU Memory Usage {:s} {:s} {:s}:".format('-' * 10, custom_msg, '-' * 10))
    logger.info(f"  Current Allocated Memory : {allocated_memory:.2f} MB")
    logger.info(f"  Maximum Allocated Memory : {max_allocated_memory:.2f} MB")
    logger.info(f"  Current Reserved Memory  : {reserved_memory:.2f} MB")
    logger.info(f"  Maximum Reserved Memory  : {max_reserved_memory:.2f} MB")


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


import pickle
import time
import re
import copy
import os
import numpy as np
import torch
import tqdm
from multiprocessing import Pool
from scipy.special import softmax

from trajflow.utils.init_objective import prepare_denoiser_data
from trajflow.utils import common_utils, motion_utils
from .submission import serialize_single_batch, save_submission_file


def deep_copy_dict(batch_dict, scores, trajs):
    batch_dict_copy = {}
    keys_to_del = ['denoiser_dict', 'encoder_output', 'denoiser_output']
    for key, val in batch_dict.items():
        if key not in keys_to_del:
            batch_dict_copy[key] = copy.deepcopy(val)
    batch_dict_copy['pred_scores'] = scores
    batch_dict_copy['pred_trajs'] = trajs
    return batch_dict_copy


def eval_one_epoch(denoiser, test_loader, cfg, epoch_id, logger,  
                   inter_pred=False, flag_submission=False, submission_info=None,
                   logger_iter_interval=50):
    # Init
    dist_test = cfg.OPT.DIST_TRAIN
    eval_output_dir = cfg.SAVE_DIR.EVAL_OUTPUT_DIR

    test_set = test_loader.dataset

    pred_dicts = []                                 # denoiser trajectory + denoiser classifer score
    scenario_predictions = []                       # submission format

    # Adjust the model for evaluation
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        if not isinstance(denoiser, torch.nn.parallel.DistributedDataParallel):
            num_gpus = torch.cuda.device_count()
            local_rank = cfg.LOCAL_RANK % num_gpus
            denoiser = torch.nn.parallel.DistributedDataParallel(
                    denoiser,
                    device_ids=[local_rank],
                    broadcast_buffers=False
            )
    denoiser.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(test_loader), leave=True, desc='eval', dynamic_ncols=True)

    # Evaluation loop
    start_time = time.time()

    for i, batch_dict in enumerate(test_loader):
        disp_dict = {}
        
        with torch.no_grad():
            """prepare data for denoiser model"""
            if 'center_gt_trajs' in batch_dict['input_dict']:
                batch_dict = prepare_denoiser_data(batch_dict, cfg.DATA_CONFIG.DATA_RESCALE, cfg.DEVICE)
            
            batch_dict['denoiser_dict'] = {}

            """create more samples in a for loop"""
            pred_trajs, pred_cls_logits, batch_dicts = denoiser(batch_dict, disp_dict=disp_dict, flag_sample=True)
            
            """use denoiser cls score for NMS"""
            pred_scores_cls_nms = batch_dicts['pred_scores']
            pred_trajs_cls_nms = batch_dicts['pred_trajs']

            batch_cls_score = deep_copy_dict(batch_dicts, pred_scores_cls_nms, pred_trajs_cls_nms)

            final_pred_dicts = test_set.generate_prediction_dicts(batch_cls_score, 
                                                                  inter_pred=inter_pred, 
                                                                  flag_submission=flag_submission)
            pred_dicts += final_pred_dicts

            if flag_submission:
                scenario_predictions.extend(serialize_single_batch(final_pred_dicts, inter_pred))

            B, K, T = pred_trajs.size()[:3]

        ### end of torch.no_grad() ###
    
        # log the evaluation results
        if cfg.LOCAL_RANK == 0 and (i % logger_iter_interval == 0 or i == 0 or i + 1== len(test_loader)):
            past_time = progress_bar.format_dict['elapsed']
            second_each_iter = past_time / max(i, 1.0)
            remaining_time = second_each_iter * (len(test_loader) - i)
            disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
            batch_size = batch_dict.get('batch_size', None)
            logger.info(f'eval: epoch={epoch_id}, batch_iter={i}/{len(test_loader)}, batch_size={batch_size}, iter_cost={second_each_iter:.2f}s, '
                        f'time_cost: {progress_bar.format_interval(past_time)}/{progress_bar.format_interval(remaining_time)}, '
                        f'{disp_str}')
    ### end of evaluation loop ###

    """eval data saving and logging"""
    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        logger.info(f'Total number of samples before merging from multiple GPUs: {len(pred_dicts)}')
        pred_dicts = common_utils.merge_results_dist(pred_dicts, len(test_set), tmpdir=os.path.join(eval_output_dir, 'tmpdir'))
        if cfg.LOCAL_RANK == 0:
            logger.info(f'Total number of samples after merging from multiple GPUs (removing duplicate): {len(pred_dicts)}')

        if flag_submission:
            scenario_predictions = common_utils.merge_results_dist(scenario_predictions, len(test_set), tmpdir=os.path.join(eval_output_dir, 'tmpdir'))

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(test_loader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}

    logger.info("Number of total trajectories to evaluate: {:d}".format(len(pred_dicts)))
    with open(os.path.join(eval_output_dir, 'result_denoiser.pkl'), 'wb') as f:
        pickle.dump(pred_dicts, f)

    if flag_submission:
        save_submission_file(scenario_predictions, inter_pred, eval_output_dir, cfg.OUTPUT_DIR_PREFIX, submission_info, logger)

    """evaluate trajectory performance"""
    def _get_latex_str(in_str):
        # extract the last line of the evaluation results and reorganize it into a latex-friendly string
        str_latex = in_str.split('\n')[-2].split(',')[:4]
        str_latex = [float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0]) for s in str_latex]
        str_latex = str_latex[1:] + [str_latex[0]]
        str_latex = ' & '.join(['{:.4f}'.format(float(s)) for s in str_latex])
        return str_latex
    
    def _eval_and_log(pred_dicts, keyword):
        if len(pred_dicts):
            result_str, result_dict = test_set.evaluation(pred_dicts, inter_pred=inter_pred)
            # logger.info('\n{} Diffusion output results {}'.format('*' * 20, '*' * 20) + result_str)
            logger.info('\n{:s} {:s} output results {:s}'.format('*' * 20, keyword, '*' * 20) + '\n'.join(result_str.split('\n')[-7:]))
            result_latex = _get_latex_str(result_str)
            logger.info('{:s} output results in latex-friendly format: '.format(keyword) + result_latex)
            result_dict = {'{:s}_'.format(keyword) + key: val for key, val in result_dict.items()}
            ret_dict.update(result_dict)
        else:
            logger.info("Skip {:s} results evaluation as no relevant results are available.".format(keyword))
    
    if test_set.mode in ['eval', 'inter_eval']:
        _eval_and_log(pred_dicts, 'denoiser')

    logger.info('Result is save to %s' % eval_output_dir)
    logger.info('****************Evaluation done.*****************')

    return ret_dict


if __name__ == '__main__':
    pass

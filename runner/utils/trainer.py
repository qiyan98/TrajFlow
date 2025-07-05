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
import glob
import os
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
from trajflow.utils.init_objective import prepare_denoiser_data
from utils.tester import eval_one_epoch
import wandb


def _get_next_batch(dataloader_iter, train_loader, logger, cur_it):
    """Get next batch, reset iterator if needed."""
    try:
        return next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(train_loader)
        logger.info(f'Resetting dataloader iterator at iter {cur_it}')
        return next(dataloader_iter)


def _log_training_info(logger, cur_epoch, total_epochs, accumulated_iter, cur_it, 
                      total_it_each_epoch, batch_size, second_each_iter, 
                      trained_time_each_epoch, remaining_second_each_epoch,
                      trained_time_past_all, remaining_second_all, disp_dict, tbar):
    """Log training information to console."""
    disp_str = ', '.join([f'{key}={val:.3f}' for key, val in disp_dict.items() if key != 'lr'])
    disp_str += f', lr={disp_dict["lr"]:.3e}'
    
    logger.info(
        f'epoch: {cur_epoch}/{total_epochs}, acc_iter={accumulated_iter}, '
        f'cur_iter={cur_it}/{total_it_each_epoch}, batch_size={batch_size}, '
        f'iter_cost={second_each_iter:.2f}s, '
        f'time_cost(epoch): {tbar.format_interval(trained_time_each_epoch)}/'
        f'{tbar.format_interval(remaining_second_each_epoch)}, '
        f'time_cost(all): {tbar.format_interval(trained_time_past_all)}/'
        f'{tbar.format_interval(remaining_second_all)}, {disp_str}')


def _log_to_wandb(wb_log, cur_lr, wb_dict, grad_total_norm, accumulated_iter):
    """Log metrics to wandb."""
    if wb_log is None:
        return
        
    # Log all metrics in a single call to ensure step consistency
    log_dict = {
        'meta_data/learning_rate': cur_lr,
        'train/grad_total_norm': grad_total_norm,
        'iteration': accumulated_iter  # Add iteration step
    }
    
    for key, val in wb_dict.items():
        if 'denoiser_loss_per_level' in key:
            # Create matplotlib figure for loss per level
            fig = plt.figure()
            t_plot = np.array(list(val.keys()))
            val_plot = np.array(list(val.values()))
            flag_valid = val_plot > 0
            plt.plot(t_plot[flag_valid], val_plot[flag_valid], '-o')
            plt.xlim(t_plot.min(), t_plot.max())
            plt.xlabel('Noise level')
            plt.ylabel('Loss')
            log_dict[f'train/{key}'] = wandb.Image(fig)
            plt.close(fig)
        else:
            log_dict[f'train/{key}'] = val
    
    # Log all metrics at once with the same step
    wb_log.log(log_dict)


def _should_save_checkpoint(trained_epoch, ckpt_save_interval, total_epochs, rank):
    """Check if checkpoint should be saved."""
    return (trained_epoch % ckpt_save_interval == 0 or 
            trained_epoch in [1, 30, 40] or 
            trained_epoch > total_epochs - 5) and rank == 0


def _should_run_eval(test_loader, trained_epoch, ckpt_save_interval, total_epochs):
    """Check if evaluation should be run."""
    return (test_loader is not None and 
            (trained_epoch % ckpt_save_interval == 0 or 
             trained_epoch in [1, 30, 40] or 
             trained_epoch > total_epochs - 10))


def _cleanup_old_checkpoints(ckpt_save_dir, max_ckpt_save_num):
    """Remove old checkpoints to maintain limit."""
    ckpt_list = glob.glob(os.path.join(ckpt_save_dir, 'checkpoint_epoch_*.pth'))
    ckpt_list.sort(key=os.path.getmtime)
    
    if len(ckpt_list) >= max_ckpt_save_num:
        for cur_file_idx in range(len(ckpt_list) - max_ckpt_save_num + 1):
            os.remove(ckpt_list[cur_file_idx])


def _get_map_score(wb_dict):
    """Extract mAP score from wandb dict."""
    if 'denoiser_mAP' in wb_dict:
        return wb_dict['denoiser_mAP']
    elif 'mAP' in wb_dict:
        return wb_dict['mAP']
    else:
        raise ValueError('No mAP in wb_dict')


def _update_best_model_record(best_record_file, trained_epoch, map_score):
    """Update best model record file."""
    try:
        with open(best_record_file, 'r') as f:
            best_src_data = f.readlines()
        best_performance = float(best_src_data[-1].strip().split(' ')[-1])
    except:
        with open(best_record_file, 'a') as f:
            pass
        best_performance = -1
        best_src_data = []

    with open(best_record_file, 'a') as f:
        print(f'epoch_{trained_epoch} mAP {map_score}', file=f)

    return best_performance, best_src_data


def train_one_epoch(denoiser, optimizer, scheduler, train_loader, ema_helper, cfg,
                    cur_epoch, accumulated_iter, logger, wb_log, tbar, 
                    leave_pbar=False, ckpt_save_time_interval=300, logger_iter_interval=50):
    """Train for one epoch."""
    rank = cfg.LOCAL_RANK
    total_epochs = cfg.OPT.NUM_EPOCHS
    ckpt_save_dir = cfg.SAVE_DIR.CKPT_DIR

    ckpt_save_cnt = 1
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)
    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)
    start_it = accumulated_iter % total_it_each_epoch

    denoiser.train()

    for cur_it in range(start_it, total_it_each_epoch):
        # Get batch data
        batch_dict = _get_next_batch(dataloader_iter, train_loader, logger, cur_it)
        cur_lr = optimizer.param_groups[0]['lr']

        # Forward pass
        optimizer.zero_grad()
        batch_dict['cur_epoch'] = cur_epoch
        batch_dict = prepare_denoiser_data(batch_dict, cfg.DATA_CONFIG.DATA_RESCALE, cfg.DEVICE)

        disp_dict, wb_dict = {}, {}
        loss_denoiser_reg, loss_denoiser_cls, batch_dict = denoiser(batch_dict, disp_dict=disp_dict, wb_dict=wb_dict)
        loss = loss_denoiser_reg + loss_denoiser_cls

        # Backward pass
        loss.backward()
        grad_total_norm = clip_grad_norm_(denoiser.parameters(), cfg.OPT.GRAD_NORM_CLIP)
        optimizer.step()

        # Update EMA and scheduler
        if ema_helper is not None:
            [ema.update() for ema in ema_helper]
        scheduler.step()

        accumulated_iter += 1
        disp_dict.update({'loss_total': loss.item(), 'lr': cur_lr})
        wb_dict.update({'loss_total': loss.item()})
        
        # Logging
        if rank == 0:
            if (accumulated_iter % logger_iter_interval == 0 or 
                cur_it == start_it or cur_it + 1 == total_it_each_epoch):
                
                # Calculate timing info
                trained_time_past_all = tbar.format_dict['elapsed']
                second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)
                trained_time_each_epoch = pbar.format_dict['elapsed']
                remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                batch_size = batch_dict.get('batch_size', None)
                _log_training_info(logger, cur_epoch, total_epochs, accumulated_iter, cur_it,
                                 total_it_each_epoch, batch_size, second_each_iter,
                                 trained_time_each_epoch, remaining_second_each_epoch,
                                 trained_time_past_all, remaining_second_all, disp_dict, tbar)

            if wb_log is not None:
                _log_to_wandb(wb_log, cur_lr, wb_dict, grad_total_norm, accumulated_iter)

            # Save checkpoint during training
            time_past_this_epoch = pbar.format_dict['elapsed']
            if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                ckpt_name = os.path.join(ckpt_save_dir, 'latest_model')
                save_checkpoint(checkpoint_state(denoiser, optimizer, ema_helper, cur_epoch, accumulated_iter), filename=ckpt_name)
                logger.info(f'Save latest model at epoch {cur_epoch} and iter {accumulated_iter} to {ckpt_name}')
                ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()

    return accumulated_iter


def train_model(denoiser, optimizer, scheduler, train_loader, ema_helper, cfg, 
                start_epoch, start_iter, logger, wb_log, train_sampler=None, test_loader=None, 
                ckpt_save_interval=1, ckpt_save_time_interval=300, max_ckpt_save_num=50, 
                logger_iter_interval=50):
    """Main training loop."""
    accumulated_iter = start_iter
    total_epochs = cfg.OPT.NUM_EPOCHS
    rank = cfg.LOCAL_RANK
    ckpt_save_dir = cfg.SAVE_DIR.CKPT_DIR
    eval_output_dir = cfg.SAVE_DIR.EVAL_OUTPUT_DIR

    # Define wandb metrics to use different step scales
    if rank == 0 and wb_log is not None:
        # Training metrics use iteration steps
        wandb.define_metric("train/*", step_metric="iteration")
        # Evaluation metrics use epoch steps
        wandb.define_metric("eval/*", step_metric="epoch")
        # Meta data can use either
        wandb.define_metric("meta_data/*", step_metric="iteration")

    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        for cur_epoch in tbar:
            torch.cuda.empty_cache()

            # Set epoch for distributed training
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            trained_epoch = cur_epoch + 1
            flag_save_ckpt = _should_save_checkpoint(trained_epoch, ckpt_save_interval, total_epochs, rank)
            flag_run_eval = _should_run_eval(test_loader, trained_epoch, ckpt_save_interval, total_epochs)

            # Train one epoch
            accumulated_iter = train_one_epoch(
                denoiser, optimizer, scheduler, train_loader, ema_helper, cfg,
                cur_epoch, accumulated_iter, logger, wb_log, tbar, 
                leave_pbar=(cur_epoch + 1 == total_epochs),
                ckpt_save_time_interval=ckpt_save_time_interval, 
                logger_iter_interval=logger_iter_interval)

            # Save checkpoint
            if flag_save_ckpt:
                _cleanup_old_checkpoints(ckpt_save_dir, max_ckpt_save_num)
                ckpt_name = os.path.join(ckpt_save_dir, f'checkpoint_epoch_{trained_epoch:03d}_iter_{accumulated_iter:06d}')
                save_checkpoint(checkpoint_state(denoiser, optimizer, ema_helper, cur_epoch, accumulated_iter), filename=ckpt_name)

            # Evaluation
            if flag_run_eval:
                logger.info("Use online (non-EMA) model for evaluation")
                torch.cuda.empty_cache()

                eval_denoiser = denoiser
                eval_denoiser.eval()
                wb_dict = eval_one_epoch(
                    eval_denoiser, test_loader, cfg, trained_epoch, logger,
                    logger_iter_interval=max(logger_iter_interval // 5, 1))

                del eval_denoiser
                torch.cuda.empty_cache()

                if rank == 0:
                    # Log to wandb
                    wb_dict = {k: v for k, v in wb_dict.items() if '-----' not in k}  # skip meaningless entries
                    eval_log_dict = {f'eval/{key}': val for key, val in wb_dict.items()}
                    eval_log_dict['epoch'] = trained_epoch
                    wb_log.log(eval_log_dict)

                    # Check best model
                    best_record_file = os.path.join(eval_output_dir, 'best_eval_record.txt')
                    map_score = _get_map_score(wb_dict)
                    best_performance, best_src_data = _update_best_model_record(best_record_file, trained_epoch, map_score)

                    if best_performance == -1 or map_score > float(best_performance):
                        ckpt_name = os.path.join(ckpt_save_dir, 'best_model')
                        save_checkpoint(checkpoint_state(denoiser, optimizer, ema_helper, cur_epoch, accumulated_iter), filename=ckpt_name)
                        logger.info(f'Save best model at epoch {trained_epoch} and mAP {map_score} to {ckpt_name}')

                        with open(best_record_file, 'a') as f:
                            print(f'best_epoch_{trained_epoch} mAP {map_score}', file=f)
                    else:
                        with open(best_record_file, 'a') as f:
                            print(f'{best_src_data[-1].strip()}', file=f)


def model_state_to_cpu(model_state):
    """Convert model state to CPU."""
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(denoiser, optimizer=None, ema_helper=None, epoch=None, it=None):
    """Create checkpoint state dictionary."""
    optim_state = optimizer.state_dict() if optimizer is not None else None

    if isinstance(denoiser, torch.nn.parallel.DistributedDataParallel):
        model_state = model_state_to_cpu(denoiser.module.model.state_dict())
    else:
        model_state = denoiser.model.state_dict()

    try:
        import trajflow
        version = 'trajflow+' + trajflow.__version__
    except:
        version = 'none'

    to_save = {
        'epoch': epoch, 'it': it, 'model_state': model_state, 
        'optimizer_state': optim_state, 'version': version
    }
    
    if ema_helper is not None:
        for ema in ema_helper:
            beta = ema.beta
            to_save[f'model_ema_beta_{beta:.4f}'] = ema.ema_model.state_dict()
    
    return to_save


def save_checkpoint(state, filename='checkpoint'):
    """Save checkpoint to file."""
    torch.save(state, f'{filename}.pth')

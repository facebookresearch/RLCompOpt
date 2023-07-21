
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import os
import pickle
import random
import sqlite3
import time
import warnings
from typing import Callable

import common_utils
import hydra
import matplotlib.pyplot as plt
import numpy as np
import shutil
import submitit
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from submitit.core.utils import FailedJobError
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import uuid
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, to_absolute_path

import rlcompopt.utils as utils
from rlcompopt.cl.dataset import (
    CompilerGymDatasetOnline,
    CompilerGymDatasetOnlineSocket,
    CompilerGymDatasetPyG,
    CompilerGymDatasetPydantic,
    dummy_collate,
)
from rlcompopt.pipeline.lib.types import TrajectoryDataset
from rlcompopt.cl.faster_balanced_sampler import BalancedBatchSampler
from rlcompopt.cl.faster_balanced_sampler_stream import BalancedStreamer
from rlcompopt.cl.models.model_utils import load_model
from rlcompopt.cl.repr_queue import ReprQueue
from rlcompopt.env_wrapper.pyg_utils import remove_edges
from rlcompopt.model_testing import test_model, get_model_rowid


warnings.filterwarnings("ignore", module="torch_geometric")  # 'error'
__this_path = os.path.abspath(os.path.dirname(__file__))


def get_dataset(
    db_file,
    batch_size,
    args,
    load_balance=False,
    train=True,
    autophase_max_bin=10,
    num_workers=32,
    output_key="reward",
    featurized_dataset=None,
    pydantic_dataset_path=None,
    log=None,
):
    if args.model.mode == "pyg":
        CompilerGymDataset = CompilerGymDatasetPyG
    kwargs = {}
    if args.model.gnn_type == "GraphAttnWin":
        kwargs["add_block_idx"] = True
    kwargs["random_mixup"] = args.dataset.random_mixup
    kwargs["seq_classification"] = args.seq_classification
    kwargs["dense_seq_cls"] = args.dataset.dense_seq_cls
    kwargs["pydantic_dataset_path"] = pydantic_dataset_path
    kwargs["dense_cls_metric"] = args.dataset.dense_cls_metric
    kwargs["action_histogram_steps"] = args.model.action_histogram_steps
    kwargs["num_actions"] = args.model.num_actions
    kwargs["remove_type_graph"] = args.dataset.remove_type_graph

    if args.generate_v4:
        if args.dataset.send_data_via_socket:
            CompilerGymDataset = CompilerGymDatasetOnlineSocket
            # total_workers = num_workers * utils.get_world_size()
            # num_socket_conn = args.dataset.num_generators // total_workers
            # assert args.dataset.num_generators == num_socket_conn * total_workers
            kwargs.update(
                dict(
                    send_data_via_socket=args.dataset.send_data_via_socket,
                    socket_db=db_file.replace("summary", "socket"),
                    num_workers=num_workers,
                    num_servers=args.dataset.num_generators,
                    bin_size=args.dataset.max_nodes,
                    full_rate=args.dataset.full_rate,
                    num_max_bin=10,
                )
            )
        else:
            CompilerGymDataset = CompilerGymDatasetOnline
            if pydantic_dataset_path:
                CompilerGymDataset = CompilerGymDatasetPydantic
            sampler = BalancedStreamer(
                db_file,
                max_size=args.dataset.max_nodes,
                full_rate=args.dataset.full_rate,
                num_replicas=utils.get_world_size() if args.distributed else 1,
                rank=utils.get_rank() if args.distributed else 0,
                circulate_data=args.dataset.circulate_data,
                eval_data_len=args.dataset.eval_data_len if not train else 0,
                num_records=args.dataset.num_records,
                exclude_sets=args.dataset.exclude_sets,
                weight_data_resample=args.dataset.weight_data_resample,
                seq_classification=args.seq_classification,
                auto_batchsize=args.dataset.auto_batchsize,
            )

    dataset = CompilerGymDataset(
        db_file,
        autophase_max_bin=autophase_max_bin,
        output_key=output_key,
        featurized_dataset=featurized_dataset,
        pre_load=args.dataset.pre_load,
        load_next_state=args.dataset.load_next_state,
        remove_large_graph=args.dataset.remove_large_graph,
        max_nodes=args.dataset.max_nodes,
        ssl=args.ssl,
        load_balance=load_balance,
        load_cumulative_reward2=args.dataset.load_cumulative_reward2,
        divided_by_this_ir=args.dataset.divided_by_this_ir,
        load_subgraph_feature=args.model.use_subgraph_feature,
        subgraph_feature=args.model.subgraph,
        use_autophase=args.model.use_autophase,
        use_history=args.model.use_history,
        use_cl=args.model.use_cl,
        queue_size=args.dataset.queue_size,
        min_queue_size=args.dataset.min_queue_size,
        graph_feat_dim=args.model.node_hidden_size,
        q_learning=args.dataset.q_learning,
        cache_data=args.dataset.cache_data,
        real_q_learning=args.dataset.real_q_learning,
        **kwargs,
    )

    if args.dataset.send_data_via_socket:
        data_loader = DataLoader(
            dataset,
            num_workers=num_workers,
            collate_fn=dummy_collate,
            prefetch_factor=1 if num_workers > 0 else 2,
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )
        return dataset, data_loader

    if utils.is_main_process() and not args.generate_v4:
        log.info(f"length of dataset is: {len(dataset)}")
    if args.distributed and not args.generate_v4:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if train:
            if load_balance:
                graph_key = dataset.graph_key
                size_func = lambda x: x[graph_key].num_nodes  # in "pyg" mode
                if not args.dataset.pre_load:
                    size_func = lambda x: x["dgl_graph_num_nodes"]
                sampler = BalancedBatchSampler(
                    dataset,
                    max_size=args.dataset.max_nodes,
                    full_rate=args.dataset.full_rate,
                    size_func=size_func,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=True,
                    drop_last=False,
                )
            else:
                sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
                )
        elif args.dist_eval:
            if len(dataset) % num_tasks != 0 and utils.is_main_process():
                log.info(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            if load_balance:
                size_func = (
                    lambda x: x["dgl_graph"].num_nodes
                    if args.model.mode == "pyg"
                    else lambda x: x["dgl_graph"].num_nodes()
                )
                sampler = BalancedBatchSampler(
                    dataset,
                    max_size=args.dataset.max_nodes,
                    size_func=size_func,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=False,
                    drop_last=False,
                )
            else:
                sampler = torch.utils.data.DistributedSampler(
                    dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    elif not args.generate_v4:
        Sampler = (
            torch.utils.data.RandomSampler
            if train
            else torch.utils.data.SequentialSampler
        )
        sampler = Sampler(dataset)
        if load_balance:
            graph_key = dataset.graph_key
            size_func = lambda x: x[graph_key].num_nodes  # in "pyg" mode
            if not args.dataset.pre_load:
                size_func = lambda x: x["dgl_graph_num_nodes"]
            sampler = BalancedBatchSampler(
                dataset,
                max_size=args.dataset.max_nodes,
                full_rate=args.dataset.full_rate,
                size_func=size_func,
                num_replicas=1,
                rank=0,
                shuffle=True,
                drop_last=False,
            )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size if not load_balance and not args.generate_v4 else 1,
        shuffle=train and sampler is None,
        sampler=sampler if not load_balance and not args.generate_v4 else None,
        batch_sampler=sampler if load_balance or args.generate_v4 else None,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn_ssl if args.ssl else dataset.collate_fn,
        drop_last=train and not load_balance and not args.generate_v4,
        timeout=args.dataset.timeout,
        prefetch_factor=2,
        pin_memory=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    return dataset, data_loader


def linear_warmup(step, warmup_steps, optimizer):
    if step > warmup_steps or optimizer is None:
        return
    for g in optimizer.param_groups:
        orig_lr = g.get("_orig_lr", None)
        if orig_lr is None:
            g["_orig_lr"] = g["lr"]
        g["lr"] = g["_orig_lr"] * float(step) / warmup_steps


@torch.no_grad()
def momentum_update(
    module_online: nn.Module, module_offline: nn.Module, momentum: float
):
    for param_q, param_k in zip(
        module_online.parameters(), module_offline.parameters()
    ):
        param_k.data = param_k.data * momentum + param_q.data * (1.0 - momentum)


def analyze(y_pred, actions, num=124):
    y_pred = torch.cat(y_pred)
    actions = torch.cat(actions)
    correct = (y_pred == actions).float()
    accs = {}
    for i in range(num):
        mask = actions == i
        cor = correct[mask]
        if cor.numel() == 0:
            continue
        acc = cor.sum() / cor.numel()
        accs[i] = acc.item()
    return accs


def analyze2(logits, actions, num=124, topk=(1,3,5)):
    logits = torch.cat(logits, dim=0)
    actions = torch.cat(actions)
    overall_acc = accuracy(logits, actions, topk=topk)
    accs = {}
    for i in range(num):
        mask = actions == i
        ll = logits[mask]
        aa = actions[mask]
        if aa.numel() == 0:
            continue
        acc = accuracy(ll, aa, topk=topk)
        accs[i] = tuple(acc)
    return overall_acc, accs


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0).item() / batch_size for k in topk]


def topk_score(output, target_score, topk=(1,)):
    """Computes the max_score for the k top predictions"""
    assert output.shape == target_score.shape
    maxk = min(max(topk), output.size()[1])
    _, pred = output.topk(maxk, 1, True, True)
    topk_scores = []
    for k in topk:
        scores = torch.gather(target_score, dim=1, index=pred[:, :k])
        max_scores = scores.max(dim=1)[0]
        topk_scores.append(max_scores.mean().item())
    return topk_scores


def policy_gradient(
    data_loader,
    model,
    model_without_ddp,
    device,
    optimizer=None,  # should be None in eval mode
    save_model: Callable = None,
    save_val_best: Callable = None,
    save_frequence: int = None,
    print_frequence: int = None,
    warmup_steps: int = None,
    model_ema: nn.Module = None,
    lr_schedular=None,  # should be None in eval mode
    total_steps: int = None,
    eval_mode=False,  # for evaluation in offline training
    eval_loader=None,  # need to call eval function if not None
    eval_frequence=None,
    args=None,
    log=None,
):
    if utils.is_main_process() and not eval_mode:
        rowid = save_model(model_without_ddp.state_dict(), model_ema)
    if args.submitit.log_dir is not None and not eval_mode:
        t0 = time.time()
    model.train(not eval_mode)
    last_loss, eval_loss = None, None

    repr_queue = ReprQueue(
        data_loader.dataset.graph_feat_dim,
        K=data_loader.dataset.queue_size,
        min_K=data_loader.dataset.min_queue_size,
    )

    def avg_over_proc(loss, batchsize):
        batchsize_ = torch.tensor([int(batchsize)], dtype=torch.long, device=device)
        dist.all_reduce(batchsize_, op=dist.ReduceOp.SUM)
        dist.barrier()
        loss = loss * (float(batchsize) * dist.get_world_size() / batchsize_.item())
        return loss, batchsize_.item()

    t1 = time.time()
    last_loss = -1
    eval_losses = []
    for iters, data in enumerate(data_loader):
        if iters % print_frequence == 0 or (
            total_steps is not None and iters >= total_steps
        ):
            if utils.is_main_process() and iters > 0:
                if eval_mode:
                    log.info(f"Eval results:")
                if args.behavior_cloning:
                    overall_acc, accs = analyze2(y_pred_, actions_)
                    log.info(f"Per action accuracy: {accs}")
                    stats["acc1"].feed(overall_acc[0])
                    stats["acc3"].feed(overall_acc[1])
                    stats["acc5"].feed(overall_acc[2])
                    logits_tmp = torch.cat(y_pred_, dim=0)
                    labels_tmp = torch.cat(labels_, dim=0)
                    k_scores = topk_score(logits_tmp, labels_tmp, topk=(1, 3, 5))
                    stats["k_scores1"].feed(k_scores[0])
                    stats["k_scores3"].feed(k_scores[1])
                    stats["k_scores5"].feed(k_scores[2])

                log.info(stats.summary(iters, multipliers=dict(loss=1000, lr=1000)))
                log.info(
                    f"Avg per sample loss at iters [{iters}]: {sum_loss / total_sample} [{total_sample}]"
                )
            if iters > 0:
                last_loss = sum_loss / total_sample
            # reset statistics
            sum_loss = 0
            total_sample = 0
            stats = common_utils.MultiCounter()
            stats.reset()
            if args.behavior_cloning:
                y_pred_ = []
                actions_ = []
                labels_ = []
            if args.submitit.log_dir is not None and not eval_mode:
                if utils.is_dist_avail_and_initialized():
                    dist.barrier()
                elapse = (time.time() - t0) / 60
                if int(args.submitit.timeout_min) - elapse < 20:
                    print("Run out of time. Exiting...")
                    log.info("Run out of time. Exiting...")
                    break
        if eval_loader is not None and iters > 0 and iters % eval_frequence == 0:
            with torch.no_grad():
                eval_loss = policy_gradient(
                    eval_loader,
                    model,
                    model_without_ddp,
                    device,
                    print_frequence=print_frequence,
                    warmup_steps=warmup_steps,
                    eval_mode=True,
                    args=args,
                    log=log,
                )
            eval_losses.append(eval_loss)
            min_eval_loss = min(eval_losses)
            min_idx = eval_losses.index(min_eval_loss)
            step_to_min = len(eval_losses) - 1 - min_idx
            if step_to_min == 0 and utils.is_main_process():
                save_val_best(rowid)
            if step_to_min * eval_frequence > 1000 and args.early_stop:
                print("Stop early.")
                total_steps = 0
            t1 = time.time()
        linear_warmup(iters + 1, warmup_steps, optimizer)
        if iters + 1 > warmup_steps and lr_schedular is not None:
            lr_schedular.step()

        if isinstance(data, list):
            data = data[0]  # get the real data tuple from IterableDataset
        # check for datastream termination (only for online training with socket)
        if args.dataset.send_data_via_socket and utils.is_dist_avail_and_initialized():
            dist.barrier()
        if data is None or (args.dataset.send_data_via_socket and data_loader.dataset.term_signal()):
            break

        (
            graph,
            actions,
            autophase,
            labels,
            advantage,
            logp,
            reward_sign_hist,
            seq_pos,
            action_history,
            reward_history,
            padding_mask,
            cl_data,
            avg_time_stamp,
        ) = data

        stats["get_data"].feed(time.time() - t1)
        stats["data_delay"].feed(time.time() - avg_time_stamp)
        t1 = time.time()

        graph = graph.to(device, non_blocking=True) if graph is not None else None
        actions = actions.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        reward_sign_hist = (
            reward_sign_hist.to(device, non_blocking=True)
            if reward_sign_hist is not None
            else None
        )
        seq_pos = seq_pos.to(device, non_blocking=True) if seq_pos is not None else None
        advantage = (
            advantage.to(device, non_blocking=True) if advantage is not None else None
        )
        logp = logp.to(device, non_blocking=True) if logp is not None else None
        autophase = (
            autophase.to(device, non_blocking=True).float()
            if autophase is not None
            else None
        )  # * 0.01
        action_history = (
            action_history.to(device, non_blocking=True)
            if action_history is not None
            else None
        )
        reward_history = (
            reward_history.to(device, non_blocking=True)
            if reward_history is not None
            else None
        )
        padding_mask = (
            padding_mask.to(device, non_blocking=True)
            if padding_mask is not None
            else None
        )
        cl_data_ = (None,) * 3
        if model_without_ddp.use_cl:
            cl_data = repr_queue.collate_fn(cl_data)
            if cl_data is not None:
                cl_data_ = [ss.to(device, non_blocking=True) for ss in cl_data]

        for local_step in range(model_without_ddp.num_local_updates):
            if model_without_ddp.use_ppo:
                if local_step == 0:
                    logp = None
                else:
                    logp = pi_info["logp_old"]
                loss_pi, loss_v, pi_info, loss_entropy = model(
                    graph, autophase, actions, advantage, labels, logp,
                    action_rewards=(action_history, reward_history),
                    padding_mask=padding_mask,
                    reward_sign_hist=reward_sign_hist,
                    seq_pos=seq_pos,
                )
                # multiply loss_pi by 0 if update_pi == False
                loss = loss_pi * float(pi_info["update_pi"]) + loss_v + loss_entropy
                stats["loss_pi"].feed(loss_pi.item())
                stats["loss_v"].feed(loss_v.item())
                stats["norm_entropy"].feed(pi_info["ent"])
                stats["clipfrac"].feed(pi_info["cf"])
                stats["update_pi"].feed(float(pi_info["update_pi"]))
                stats["log_barrier"].feed(pi_info["log_barrier"])
            elif model_without_ddp.online_q_learning:
                loss, _, abs_err, rel_err, q_value = model(
                    graph,
                    None,
                    actions,
                    labels,
                    None,
                    None,
                    autophase,
                    None,
                )
                stats["q_loss"].feed(loss.item())
                stats["abs_err"].feed(abs_err.item())
                stats["rel_err"].feed(rel_err.item())
                stats["q_value"].feed(q_value.mean().item())
                stats["q_target"].feed(labels.mean().item())
            elif args.behavior_cloning:
                loss, kl, y_pred, logits = model(
                    graph,
                    None,
                    actions,
                    labels,
                    None,
                    None,
                    autophase,
                    None,
                )
                stats["loss"].feed(loss.item())
                if kl is not None:
                    stats["kl"].feed(kl)
                y_pred_.append(logits.detach())
                actions_.append(actions)
                labels_.append(labels)
            else:
                loss, policy_loss, loss_entropy, normalize_entropy, cl_loss = model(
                    graph,
                    autophase,
                    actions,
                    labels,
                    action_rewards=(action_history, reward_history),
                    padding_mask=padding_mask,
                    reward_sign_hist=reward_sign_hist,
                    seq_pos=seq_pos,
                    buffer=cl_data_[0],
                    curr_repr_idx=cl_data_[1],
                    next_repr_idx=cl_data_[2],
                )

                stats["policy_loss"].feed(policy_loss.item())
                stats["loss_entropy"].feed(loss_entropy.item())
                stats["normalize_entropy"].feed(normalize_entropy.item())
                stats["cl_loss"].feed(cl_loss)

            loss_item = loss.clone().detach()
            bz = labels.shape[0]
            batchsize_sum = bz
            if utils.is_dist_avail_and_initialized():
                # need to average over processes according to #nodes/#edges (cannot be DGL graph in distributed mode)
                loss, batchsize_sum = avg_over_proc(loss, bz)
                loss_item = loss_item * bz
                dist.all_reduce(loss_item, op=dist.ReduceOp.SUM)
                loss_item = loss_item / batchsize_sum
                dist.barrier()

            num_states = (
                autophase.shape[0]
                if model_without_ddp.use_autophase
                else graph.num_graphs
            )
            stats["return"].feed(labels.detach().mean().item())
            stats["num_states"].feed(num_states)
            stats["batch_size"].feed(labels.size(0))
            stats["loss"].feed(loss_item.item())
            if graph is not None:
                stats["num_nodes"].feed(graph.num_nodes / graph.num_graphs)
            if not eval_mode:
                stats["lr"].feed(optimizer.param_groups[0]["lr"])

            sum_loss += (loss_item * batchsize_sum).detach().item()
            total_sample += batchsize_sum

            stats["model_forward"].feed(time.time() - t1)
            t1 = time.time()
            if eval_mode:
                break

            optimizer.zero_grad()
            loss.backward()
            # grad_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400.0)
            # stats["grad_clip"].feed(grad_clip.detach().item())
            optimizer.step()

            if model_ema is not None:
                # ema
                momentum_update(model_without_ddp, model_ema, model_ema.ema_momentum)

            stats["model_backward"].feed(time.time() - t1)
            t1 = time.time()

        if eval_mode and total_steps is None:
            total_steps = len(data_loader) // batchsize_sum + 1
        if (
            not eval_mode
            and iters > 0
            and ((iters % save_frequence == 0) or (total_steps is not None and iters >= total_steps))
            and utils.is_main_process()
        ):
            try:
                rowid = save_model(model_without_ddp.state_dict(), model_ema)
            except Exception as e:
                log.info(f"Failed to save model: {e}")
        t1 = time.time()
        if total_steps is not None and iters >= total_steps:
            break
    if utils.is_main_process():
        log.info("Finishing...")
    if eval_mode:
        return last_loss
    else:
        return last_loss, eval_loss


def dataset_looper(
    epoch_num,
    data_loader,
    model,
    model_without_ddp,
    device,
    optimizer=None,
    update_frequence=1,
    train=True,
    prev_iters=0,
    freeze_backbone=False,
    log=None,
    **kwargs,
):
    model.train(train)
    T = 10

    train_eval_label = "train" if train else "eval"

    stats = common_utils.MultiCounter()
    stats.reset()

    def avg_over_proc(loss, batchsize):
        batchsize_ = torch.tensor([int(batchsize)], dtype=torch.long, device=device)
        dist.all_reduce(batchsize_, op=dist.ReduceOp.SUM)
        dist.barrier()
        loss = loss * (float(batchsize) * dist.get_world_size() / batchsize_.item())
        return loss, batchsize_.item()

    sum_loss = 0
    sum_loss_manual = 0
    total_sample = 0
    # prev_iters = len(data_loader) * epoch_num

    t1 = time.time()
    for iters, data in enumerate(data_loader):
        # if data is None:
        #     continue
        (
            graph,
            actions,
            autophase,
            labels,
            raw_rewards,
            non_terminal,
            pred_labels,
            next_state_graph,
        ) = data

        stats["get_data"].feed(time.time() - t1)
        t1 = time.time()

        graph = graph.to(device, non_blocking=True) if graph is not None else None
        actions = actions.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if next_state_graph is not None:
            next_state_graph = next_state_graph.to(device, non_blocking=True)
        raw_rewards = (
            raw_rewards.to(device, non_blocking=True)
            if raw_rewards is not None
            else None
        )
        non_terminal = (
            non_terminal.to(device, non_blocking=True)
            if non_terminal is not None
            else None
        )
        autophase = (
            autophase.to(device, non_blocking=True).float()
            if autophase is not None
            else None
        )  # * 0.01
        if pred_labels is not None:
            pred_labels = (
                pred_labels.to(device, non_blocking=True)
                if pred_labels is not None
                else None
            )

        if train:
            loss, expected_state_action_values, abs_error, rel_error, Q = model(
                graph,
                next_state_graph,
                actions,
                labels,
                raw_rewards,
                non_terminal,
                autophase,
                freeze_backbone=freeze_backbone,
            )
        else:
            with torch.no_grad():
                loss, expected_state_action_values, abs_error, rel_error, Q = model(
                    graph,
                    next_state_graph,
                    actions,
                    labels,
                    raw_rewards,
                    non_terminal,
                    autophase,
                    freeze_backbone=freeze_backbone,
                )

        loss_item = loss.clone().detach()
        bz = labels.shape[0]
        batchsize_sum = bz
        if utils.is_dist_avail_and_initialized():
            # need to average over processes according to #nodes/#edges (cannot be DGL graph in distributed mode)
            loss, batchsize_sum = avg_over_proc(loss, bz)
            loss_item = loss_item * bz
            dist.all_reduce(loss_item, op=dist.ReduceOp.SUM)
            loss_item = loss_item / batchsize_sum
            dist.barrier()

        stats["return"].feed(labels.detach().mean().item())
        stats["Q"].feed(Q.detach().mean().item())
        for a in actions:
            stats["labels"].feed(a.detach().item())
        stats["batch_size"].feed(actions.size(0))
        stats["loss"].feed(loss_item.item())
        stats["expected_Q"].feed(expected_state_action_values.detach().mean().item())
        stats["rel_error"].feed(rel_error.detach().item())
        stats["abs_error"].feed(abs_error.detach().item())
        stats["lr"].feed(optimizer.param_groups[0]["lr"])

        if pred_labels is not None:
            loss_manual = (labels - pred_labels).pow(2).mean().item()
            stats["loss_manual"].feed(loss_manual)
        else:
            loss_manual = 0

        sum_loss += (loss_item * batchsize_sum).detach().item()
        sum_loss_manual += loss_manual * actions.size(0)
        total_sample += batchsize_sum

        # unscaled_mse.append(unscaled.cpu().data.numpy())
        # rel_errors.append(rel_error.cpu().data.numpy())

        stats["model_forward"].feed(time.time() - t1)

        t1 = time.time()
        # log.info("unscaled loss is: ", unscaled)

        if train:
            optimizer.zero_grad()
            loss.backward()
            # grad_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=400.0)
            # stats["grad_clip"].feed(grad_clip.detach().item())
            optimizer.step()

            stats["model_backward"].feed(time.time() - t1)
            t1 = time.time()
            if (iters + prev_iters) % update_frequence == 0:
                model_without_ddp.update_target_nets()

    # avg_loss, avg_unscaled, avg_grad_clip, rel_errors = np.mean(losses), np.mean(unscaled_mse), np.mean(epoch_grad_clip), np.mean(rel_errors)
    # log.info(f"Epoch num {epoch_num} training {train} took: {times}, loss: {avg_loss}, unscaled: {avg_unscaled}, rel_errors: {rel_errors}, grad_clip {avg_grad_clip}")
    if utils.is_main_process():
        log.info(
            stats.summary(epoch_num, multipliers=dict(loss=1000, loss_manual=1000))
        )
        log.info(
            f"Avg per sample loss [{train_eval_label}]: {sum_loss / total_sample} [{total_sample}]"
        )
        log.info(
            f"Avg per sample loss_manual [{train_eval_label}]: {sum_loss_manual / total_sample} [{total_sample}]"
        )
    return sum_loss / total_sample, iters + prev_iters
    # return avg_loss, avg_unscaled, avg_grad_clip


def dataset_looper_ssl(
    epoch_num,
    data_loader,
    model,
    model_without_ddp,
    device,
    optimizer=None,
    train=True,
    mode="dgl",
    vocab_size=0,
    rm_edges_perct=0.2,
    rm_nodes_perct=0.2,
    args=None,
    log=None,
    **kwargs,
):
    model.train(train)
    t1 = time.time()
    T = 10

    train_eval_label = "train" if train else "eval"

    stats = common_utils.MultiCounter()
    stats.reset()

    criterion = nn.BCELoss(reduction="none")
    criterion_node = nn.CrossEntropyLoss()
    # auroc = AUROC(pos_label=1)

    def avg_over_proc(loss, batchsize):
        batchsize_ = torch.tensor([int(batchsize)], dtype=torch.long, device=device)
        dist.all_reduce(batchsize_, op=dist.ReduceOp.SUM)
        dist.barrier()
        loss = loss * (float(batchsize) * dist.get_world_size() / batchsize_.item())
        return loss

    sum_loss = 0
    sum_loss_manual = 0
    total_sample = 0
    if utils.is_main_process():
        log.info(f"Start {'training' if train else 'eval'} for epoch {epoch_num}")

    for it, data in enumerate(data_loader):
        if data is None:
            continue
        graph = data

        stats["get_data"].feed(time.time() - t1)
        t1 = time.time()

        graph = graph.to(device, non_blocking=True)

        # randomly remove rm_edges_perct*100% edges
        num_edges = graph.num_edges() if mode == "dgl" else graph.num_edges
        num_removed0 = int(num_edges * rm_edges_perct)
        if mode == "dgl":
            u, v = graph.find_edges(list(range(num_edges)))

            removed_edge_ids = torch.randperm(num_edges, device=device)[:num_removed0]
            graph.remove_edges(removed_edge_ids)
            u_, v_ = u[removed_edge_ids], v[removed_edge_ids]
        else:
            ids = torch.randperm(num_edges, device=device)
            removed_edge_ids = ids[:num_removed0]
            edge_ids_to_keep = ids[num_removed0:]
            edge_index = graph.edge_index
            u_, v_ = torch.split(
                edge_index[:, removed_edge_ids], split_size_or_sections=1, dim=0
            )
            u_, v_ = u_.flatten(), v_.flatten()
            graph = remove_edges(
                graph, edge_ids_to_keep=edge_ids_to_keep, edge_attr=("flow", "position")
            )

        # randomly remove rm_nodes_perct*100% nodes
        num_nodes = graph.num_nodes if mode == "pyg" else graph.num_nodes()
        num_removed = int(num_nodes * rm_nodes_perct)
        if num_removed > 0:
            removed_node_ids = torch.randperm(num_nodes, device=device)[:num_removed]
            text_idx = (
                graph["x"] if mode == "pyg" else graph.nodes[None].data["text_idx"]
            )
            masked_node_labels = text_idx[removed_node_ids].clone()
            text_idx[removed_node_ids] = vocab_size + 1  # the index for masking nodes
        else:
            removed_node_ids = None

        u_feat, v_feat, masked_nodes_feat = model(
            graph,
            get_node_src_ids=u_,
            get_node_dst_ids=v_,
            get_masked_nodes=removed_node_ids,
        )

        if masked_nodes_feat is not None:
            pred_node_logits = model_without_ddp.node_predictor(masked_nodes_feat)
            node_loss = criterion_node(pred_node_logits, masked_node_labels.flatten())
            if args.ssl_config.use_node_type_loss:
                node_prob = pred_node_logits.softmax(-1)
                type_prob = (
                    node_prob @ data_loader.translation_table
                )  # num_node x num_type
                target_prob = torch.gather(
                    type_prob, -1, graph["type"][removed_node_ids].clone()
                )
                type_loss = torch.mean(-target_prob.log())

        pos_pairs = torch.cat([u_feat, v_feat], dim=1)
        neg_pairs = torch.cat(
            [v_feat, u_feat], dim=1
        )  # FIXME: there could exist reverse edges

        perm = torch.randperm(num_removed0, device=device)
        # v_2 = v_[perm]
        # mask = ~graph.has_edges_between(v_2, u_)  # change (u_, v_2) to (v_2, u_), so a bug is fixed here
        # if mask.any():
        #     neg_pairs2 = torch.cat([v_feat[perm][mask], u_feat[mask]], dim=1)  # add '[perm]', so a bug is fixed here
        # FIXME: there could exist edges for the shuffled node pairs
        neg_pairs2 = torch.cat([v_feat[perm], u_feat], dim=1)
        neg_pairs = torch.cat([neg_pairs, neg_pairs2])

        pairs = torch.cat([pos_pairs, neg_pairs])
        target = torch.zeros(pairs.shape[0], device=device)
        target[: pos_pairs.shape[0]] = 1.0

        pred = model_without_ddp.predictor(pairs).flatten()
        raw_loss = criterion(pred, target)
        loss = raw_loss.mean()

        auc = roc_auc_score(target.long().cpu().numpy(), pred.detach().cpu().numpy())
        raw_loss = raw_loss.detach()
        pos_loss = raw_loss[: pos_pairs.shape[0]].mean()
        neg_loss = raw_loss[pos_pairs.shape[0] :].mean()

        if masked_nodes_feat is not None:
            stats["loss_edge"].feed(loss.detach().item())
            stats["loss_node"].feed(node_loss.detach().item())
            if not utils.is_dist_avail_and_initialized():
                loss = loss + node_loss

        loss_item = loss.clone().detach()
        if utils.is_dist_avail_and_initialized():
            # need to average over processes according to #nodes/#edges (cannot be DGL graph in distributed mode)
            loss = avg_over_proc(loss, batchsize=num_removed0 * 3)
            if masked_nodes_feat is not None:
                node_loss = avg_over_proc(node_loss, batchsize=num_removed)
                loss = loss + node_loss
                if args.ssl_config.use_node_type_loss:
                    stats["type_loss"].feed(type_loss.item())
                    type_loss = avg_over_proc(type_loss, batchsize=num_removed)
                    loss = loss + type_loss * 20
            dist.all_reduce(loss_item, op=dist.ReduceOp.SUM)
            dist.barrier()
            loss_item = loss_item / dist.get_world_size()

        stats["loss"].feed(loss_item.item())
        stats["pos_loss"].feed(pos_loss.detach().item())
        stats["neg_loss"].feed(neg_loss.detach().item())
        stats["auc"].feed(float(auc))
        stats["batch_nodes"].feed(pred.size(0))

        sum_loss += loss.detach().item() * pred.size(0)
        total_sample += pred.size(0)

        stats["model_forward"].feed(time.time() - t1)

        t1 = time.time()

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            stats["model_backward"].feed(time.time() - t1)
            t1 = time.time()
        torch.cuda.synchronize()

    if not train and utils.is_main_process():
        log.info("eval==========================eval")
    if utils.is_main_process():
        log.info(
            stats.summary(epoch_num, multipliers=dict(loss=1000, loss_manual=1000))
        )
        log.info(
            f"Avg per sample loss [{train_eval_label}]: {sum_loss / total_sample} [{total_sample}]"
        )
        log.info(
            f"Avg per sample loss_manual [{train_eval_label}]: {sum_loss_manual / total_sample} [{total_sample}]"
        )
    return sum_loss / total_sample, it


def train(
    train_data_loader,
    dev_data_loader,
    model,
    model_without_ddp,
    optimizer,
    lr_schedular,
    device,
    vocab_size,
    args,
    freeze_backbone,
    log=None,
):
    if utils.is_main_process():
        log.info(f"training length is: {len(train_data_loader)}")
    if dev_data_loader is not None and utils.is_main_process():
        log.info(f"dev length is: {len(dev_data_loader)}")

    data_looper = dataset_looper_ssl if args.ssl else dataset_looper
    loss_ = []
    prev_iters = 0

    if freeze_backbone:
        model_without_ddp.freeze_backbone()
    else:
        model_without_ddp.unfreeze_backbone()

    for epoch in range(args.start_epoch, args.num_epoch):
        if args.distributed:
            sampler = (
                train_data_loader.batch_sampler
                if isinstance(train_data_loader.batch_sampler, BalancedBatchSampler)
                else train_data_loader.sampler
            )
            sampler.set_epoch(epoch)
            if dev_data_loader is not None:
                sampler = (
                    dev_data_loader.batch_sampler
                    if isinstance(dev_data_loader.batch_sampler, BalancedBatchSampler)
                    else dev_data_loader.sampler
                )
                sampler.set_epoch(epoch)
        start_time = time.time()
        loss, prev_iters = data_looper(
            epoch,
            train_data_loader,
            model,
            model_without_ddp,
            device,
            optimizer,
            vocab_size=vocab_size,
            mode=args.model.mode,
            update_frequence=args.model.update_frequence,
            rm_edges_perct=args.ssl_config.rm_edges_perct,
            rm_nodes_perct=args.ssl_config.rm_nodes_perct,
            prev_iters=prev_iters,
            freeze_backbone=freeze_backbone,
            args=args,
            log=log,
        )
        # data_looper(epoch, dev_data_loader, model, model_without_ddp, device, train=False, rm_edges_perct=args.ssl_config.rm_edges_perct, rm_nodes_perct=args.ssl_config.rm_nodes_perct)
        loss_.append(float(loss))

        if lr_schedular is not None:
            lr_schedular.step()

        if (
            epoch % args.save_per_epoch == 0 or epoch == args.num_epoch - 1
        ) and utils.is_main_process():
            torch.save(model_without_ddp, f"model_epoch{epoch}.pthw")

        if utils.is_main_process():
            log.info(f"took: {time.time() - start_time} for an epoch")
    return loss_

def print_env():
    for key in sorted(os.environ.keys()):
        if not (
            key.startswith(("SLURM_", "SUBMITIT_"))
            or key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
        ):
            continue
        value = os.environ[key]
        print(f"{key}={value}")


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/private/home/").is_dir():
        p = Path(f"/private/home/{user}/experiments")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def run_main_in_slurm(args):
    import warnings
    warnings.filterwarnings("ignore", module="torch_geometric")
    import submitit

    job_env = submitit.JobEnvironment()
    # args.output_dir = Path(str(args.output_dir).replace("%j", str(job_env.job_id)))
    args.gpu = job_env.local_rank
    args.rank = job_env.global_rank
    args.world_size = job_env.num_tasks
    if args.dataset.pydantic_dataset_path_test is not None:
        # need to start another process for testing.
        # cannot do testing within the trainer process,
        # as the cuda context has been initialized there
        args.world_size = job_env.num_tasks - 1
        if job_env.global_rank == job_env.num_tasks - 1:
            print(f"Running testing. {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
            testing(args)
            return
    print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
    main_real(args)


def submit_main():
    assert isinstance(all_args, list)
    if len(all_args) == 0:
        return
    args = all_args[0]
    assert args.submitit.log_dir is not None
    num_jobs = len(all_args)
    tasks_per_node = args.submitit.gpus_per_node
    time_prep_data = 10
    timeout_min = max(args.submitit.timeout_min, 5) + time_prep_data
    cpus_per_task = args.submitit.cpus_per_task
    if args.dataset.pydantic_dataset_path_test is not None:
        timeout_min += 15
        tasks_per_node += 1  # one for testing
        cpus_per_task = (cpus_per_task * args.submitit.gpus_per_node) // tasks_per_node
    assert timeout_min < 72 * 60
    print(f"{timeout_min=}, {num_jobs=}")
    executor = submitit.AutoExecutor(folder=args.submitit.log_dir)
    executor.update_parameters(
        cpus_per_task=cpus_per_task,
        timeout_min=timeout_min,
        slurm_partition=args.submitit.partition,
        mem_gb=args.submitit.mem_gb,
        gpus_per_node=args.submitit.gpus_per_node,
        tasks_per_node=tasks_per_node,  # one task per GPU
        nodes=1,
        slurm_constraint=args.submitit.constraint,
        slurm_signal_delay_s=120,
    )
    executor.update_parameters(name="rlcompopt")
    slurm_jobs = executor.map_array(run_main_in_slurm, all_args)

    print(f"Submitted job_id: {slurm_jobs[0].job_id}")
    # print(f"Logs and checkpoints will be saved at: {args.output_dir}")
    if args.model.use_reinforce or args.model.use_ppo:
        # non blocking
        return
    for job in slurm_jobs:
        try:
            print(job.results())
        except (FailedJobError, RuntimeError) as e:
            print(f"Job {job.job_id} failed: {e}. Info: {job.get_info()}")


@hydra.main(config_path="cl/conf/model", config_name="attn.yaml")
def main(args):
    # get the real out_dir and override some args
    hconfig = HydraConfig.get()
    outd = os.getcwd()
    # job_num = hconfig.job.num
    print(f"output_dir={outd}")
    args.model_db_path = os.path.join(outd, "model.db")
    args.outdir = outd

    # set up path properly, convert paths to absolute paths
    if args.dataset.train is not None:
        args.dataset.train = to_absolute_path(args.dataset.train)
    if args.dataset.dev is not None:
        args.dataset.dev = to_absolute_path(args.dataset.dev)
    args.dataset.vocab = to_absolute_path(args.dataset.vocab)
    if args.dataset.pydantic_dataset_path is not None:
        args.dataset.pydantic_dataset_path = to_absolute_path(args.dataset.pydantic_dataset_path)
    if args.dataset.pydantic_dataset_path_dev is not None:
        args.dataset.pydantic_dataset_path_dev = to_absolute_path(args.dataset.pydantic_dataset_path_dev)
    if args.dataset.pydantic_dataset_path_test is not None:
        args.dataset.pydantic_dataset_path_test= to_absolute_path(args.dataset.pydantic_dataset_path_test)

    if args.submitit.log_dir is not None:
        args.dist_url = get_init_file().as_uri()
        # save args to submit them in job array
        # this is especially useful in hydra multirun
        all_args.append(args)

    else:
        main_real(args)


def main_real(args):
    # print_env()
    # print("exporting PyTorch distributed environment variables")
    # dist_env = submitit.JobEnvironment()
    # print(f"{dist_env=}")
    # print_env()
    import logging
    out_dir = args.outdir or args.model_db_path.replace("model.db", "")
    log_file = os.path.join(out_dir, 'train.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    log = logging.getLogger(__name__)

    # save args
    if utils.is_main_process():
        with open(os.path.join(out_dir, 'args.pkl'), "wb") as f:
            pickle.dump(args, f)

    work_dir = os.path.abspath(os.path.dirname(__file__))
    print(f"{work_dir=}")
    if args.distributed:
        utils.init_distributed_mode(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    if utils.is_main_process():
        log.info(common_utils.print_info(args))
        log.info(common_utils.pretty_print_args(args))

    # output_key = "ir_instruction_count_oz_reward"
    output_key = "cumulative_reward"
    reward_dim = 1
    if output_key == "autophase":
        reward_dim = 56
    
    if args.dataset.cp_db_to_mem:
        if utils.is_main_process():
            shutil.copyfile(args.dataset.train, "/dev/shm/summary.db")
            shutil.copyfile(args.dataset.dev, "/dev/shm/dev.db")
        else:
            orig_size = os.path.getsize(args.dataset.dev)
            while (not os.path.exists("/dev/shm/dev.db")) or os.path.getsize("/dev/shm/dev.db") < orig_size:
                time.sleep(2)
            print("Done file copying")
        args.dataset.train = "/dev/shm/summary.db"
        args.dataset.dev = "/dev/shm/dev.db"

    vocab_db = args.dataset.vocab
    train_db = args.dataset.train
    eval_db = args.dataset.dev
    if utils.is_main_process():
        log.info(f"vocab_db: {vocab_db}")
        log.info(f"train_db: {train_db}")
        log.info(f"eval_db: {eval_db}")

    connection = sqlite3.connect(
        vocab_db, timeout=20
    )  # may need to wait for the release of db lock from other connections
    cursor = connection.cursor()
    vocab_size = int(cursor.execute(f"select count(*) from Vocabs;").fetchone()[0])
    if utils.is_main_process():
        log.info(f"Vocab size: {vocab_size}")
    connection.close()

    kwargs = dict(
        node_vocab_size=vocab_size + 2,  # +1 for unknown token, +2 for masked nodes
        reward_dim=reward_dim,
        ssl=args.ssl,
        divided_by_this_ir=args.dataset.divided_by_this_ir,
    )
    if args.dataset.pydantic_dataset_path is not None:
        traj_dataset = TrajectoryDataset.load(args.dataset.pydantic_dataset_path)
        args.model.num_actions = len(traj_dataset.action_sequences.actionseqs)

    args.model.random_mixup = args.dataset.random_mixup  # override it for command line simplicity
    model = hydra.utils.instantiate(args.model, **kwargs)

    if args.model.use_cl:
        model_ema = copy.deepcopy(model)
        model_ema = model_ema.to(device)
    else:
        model_ema = None

    save_val_best = lambda x: None
    # for saving model to database
    if args.model.on_policy_gradient:
        while True:
            try:
                model_connection = sqlite3.connect(args.model_db_path, timeout=3200)
                break
            except Exception as e:
                print(
                    f"Failed to connect to database {args.model_db_path}: {e}. Retrying..."
                )
                time.sleep(2)

        model_cursor = model_connection.cursor()
        with open(
            os.path.join(__this_path, "env_wrapper/database_model.sql"),
            "r",
        ) as f:
            DB_CREATION_SCRIPT = f.read()
        model_cursor.executescript(DB_CREATION_SCRIPT)
        model_connection.commit()
        model_config = pickle.dumps(args.model)
        model_kwargs = pickle.dumps(kwargs)

        def save_model(state_dict, model_ema):
            state_dict = {k: v.cpu() for k, v in state_dict.items()}
            state_dict_ = pickle.dumps(state_dict)
            if model_ema is not None:
                ema_state_dict = model_ema.state_dict()
                ema_state_dict = {k: v.cpu() for k, v in ema_state_dict.items()}
            else:
                ema_state_dict = None
            ema_state_dict = pickle.dumps(ema_state_dict)
            model_cursor.execute(
                "INSERT INTO Models VALUES (?, ?, ?, ?)",
                (model_config, model_kwargs, state_dict_, ema_state_dict),
            )
            rowid = model_cursor.execute("select rowid from Models order by rowid desc limit 1").fetchone()[0]
            model_connection.commit()
            return rowid
        
        def save_val_best(row_id):
            model_cursor.execute("INSERT INTO ValBest VALUES (?)", (row_id,))
            model_connection.commit()

        # if there are models saved, load the last one
        if args.eval_model_rowid:
            rec = list(
                model_cursor.execute(
                    "SELECT rowid, state_dict FROM Models where rowid = ?",
                    (int(args.eval_model_rowid), )
                )
            )
        else:
            rec = list(
                model_cursor.execute(
                    f"SELECT rowid, state_dict FROM Models ORDER BY rowid DESC LIMIT 1"
                )
            )
        if rec:
            state_dict = pickle.loads(rec[0][1])
            msg = model.load_state_dict(state_dict)
            log.info(f"Loaded the model from database row {rec[0][0]}: {msg}")

        if args.load_model_db is not None:
            load_model(model, args.load_model_db)

    if args.model.on_policy_gradient and not utils.is_main_process():
        save_model = lambda x: None

    if (
        args.finetune.ckpt is not None and not args.finetune.skip_ckpt
    ) or args.load_ckpt is not None:
        ckpt_path = args.load_ckpt or args.finetune.ckpt
        ckpt = torch.load(ckpt_path, map_location="cpu")
        assert ckpt.graph_version == args.model.graph_version
        ckpt = ckpt.state_dict()
        cc = set(ckpt.keys())
        model_state = model.state_dict()
        mm = set(model_state.keys())
        updated_keys = [k for k in mm if k in cc]
        updated = {k: ckpt[k] if k in cc else model_state[k] for k in mm}
        msg = model.load_state_dict(updated, strict=False)
        if utils.is_main_process():
            log.info(f"loaded checkpoint from {ckpt_path}: {msg}")
            log.info(f"loaded parameters: {updated_keys}")
    model = model.to(device)
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if utils.is_main_process():
        log.info(f"Number of params: {n_parameters}")
        log.info(model)

    if args.finetune.ckpt is not None:
        # 1st stage finetuning: the heads parameters
        params = model_without_ddp.head_params()
        optimizer = torch.optim.Adam(
            params, lr=args.finetune.stage1.lr, weight_decay=args.finetune.stage1.wd
        )
        args.num_epoch = args.finetune.stage1.epochs
        if utils.is_main_process():
            log.info(f"Finetune stage 1 started")
        lr = args.finetune.stage1.lr
        freeze_backbone = True
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.optim.lr, weight_decay=args.optim.weight_decay
        )
        lr = args.optim.lr
        freeze_backbone = False
    lr_schedular = None
    if args.optim.lr_schedular:
        lr_schedular = CosineAnnealingLR(
            optimizer, T_max=args.num_epoch, eta_min=lr * 0.1, verbose=True
        )

    train_dataset, train_dataset_loader = get_dataset(
        train_db,
        batch_size=args.train_batch_size,
        load_balance=args.dataset.load_balance,
        args=args,
        train=True,
        autophase_max_bin=args.dataset.autophase_max_bin,
        num_workers=args.dataset.num_workers,
        output_key=output_key,
        pydantic_dataset_path=args.dataset.pydantic_dataset_path,
        log=log,
    )
    _, dev_dataset_loader = (
        None,
        None,
    )  # get_dataset(eval_db, batch_size=args.eval_batch_size, args=args, train=False, autophase_max_bin=args.dataset.autophase_max_bin, num_workers=args.dataset.num_workers, output_key = output_key, featurized_dataset=train_dataset)
    args_ = copy.deepcopy(args)
    args_.dataset.random_mixup = 0.
    args_.dataset.weight_data_resample = False
    if args.dataset.eval_data_len > 0:
        _, dev_dataset_loader = get_dataset(
            train_db,  # share database with train set but use a different split in the database
            batch_size=args.train_batch_size,
            load_balance=args.dataset.load_balance,
            args=args_,
            train=False,
            autophase_max_bin=args.dataset.autophase_max_bin,
            num_workers=args.dataset.num_workers,
            output_key=output_key,
            pydantic_dataset_path=args.dataset.pydantic_dataset_path,
            log=log,
        )
    if args.dataset.dev is not None:
        _, dev_dataset_loader = get_dataset(
            eval_db,
            batch_size=args.train_batch_size,
            load_balance=args.dataset.load_balance,
            args=args_,
            train=False,
            autophase_max_bin=args.dataset.autophase_max_bin,
            num_workers=args.dataset.num_workers,
            output_key=output_key,
            pydantic_dataset_path=args.dataset.pydantic_dataset_path_dev,
            log=log,
        )

    if args.ssl:
        table = train_dataset.get_translation_matrix(vocab_size + 1)
        table = table.to(device)
        train_dataset_loader.translation_table = table

    if args.model.on_policy_gradient:
        if args.optim.lr_schedular:
            if not args.optim.lr_schedular_steps:
                args.optim.lr_schedular_steps = args.total_steps
            assert args.optim.lr_schedular_steps
            assert args.optim.lr_schedular_steps - args.warmup_steps > 0
            lr_schedular = CosineAnnealingLR(
                optimizer,
                T_max=args.optim.lr_schedular_steps - args.warmup_steps,
                eta_min=lr * 0.01,
                verbose=False,
            )
        if args.eval_model_rowid:
            with torch.no_grad():
                eval_loss = policy_gradient(
                    dev_dataset_loader,
                    model,
                    model_without_ddp,
                    device,
                    print_frequence=args.print_frequence,
                    warmup_steps=args.warmup_steps,
                    eval_mode=True,
                    args=args,
                    log=log,
                )
            return
        last_loss, eval_loss = policy_gradient(
            train_dataset_loader,
            model,
            model_without_ddp,
            device,
            optimizer,
            save_model,
            save_val_best,
            args.save_frequence,
            args.print_frequence,
            args.warmup_steps,
            model_ema,
            lr_schedular,
            args.total_steps,
            eval_loader=dev_dataset_loader,
            eval_frequence=args.eval_frequence,
            args=args,
            log=log,
        )
        dist.barrier()
        torch.cuda.empty_cache()
        if utils.is_main_process() and args.dataset.exclude_sets is not None:
            model_root = os.path.dirname(args.model_db_path)
            result = f"{args.dataset.exclude_sets},{last_loss:.8f},{eval_loss:.8f}\n"
            with open(os.path.join(model_root, "final_stat.txt"), "a") as f:
                f.write(result)
        if args.submitit.log_dir is not None and args.dataset.pydantic_dataset_path_test is not None:
            write_signal(args.model_db_path)  # signal the test process to start
            while get_signal(args.model_db_path) <= args.world_size:
                time.sleep(10)  # wait for the test process to finish
        if utils.is_main_process():
            log.info("Exiting trainer...")
        return

    loss = train(
        train_dataset_loader,
        dev_dataset_loader,
        model,
        model_without_ddp,
        optimizer,
        lr_schedular,
        device,
        vocab_size,
        args,
        freeze_backbone,
        log=log,
    )

    # 2nd stage finetuning: the whole model
    if args.finetune.ckpt is not None:
        args.start_epoch = args.finetune.stage1.epochs
        args.num_epoch = args.finetune.stage1.epochs + args.finetune.stage2.epochs
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.finetune.stage2.lr,
            weight_decay=args.finetune.stage2.wd,
        )
        lr_schedular = None
        if args.optim.lr_schedular:
            lr_schedular = CosineAnnealingLR(
                optimizer,
                T_max=args.num_epoch,
                eta_min=args.finetune.stage2.lr * 0.1,
                verbose=True,
            )
        if utils.is_main_process():
            log.info("Finetune stage 2 started")
        loss += train(
            train_dataset_loader,
            dev_dataset_loader,
            model,
            model_without_ddp,
            optimizer,
            lr_schedular,
            device,
            vocab_size,
            args,
            freeze_backbone=False,
            log=log,
        )
    # if isinstance(train_dataset_loader.batch_sampler, BalancedBatchSampler):
    #     train_dataset_loader.batch_sampler.worker.terminate()
    if utils.is_main_process():
        plt.plot(loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig("loss.jpg")


def write_signal(model_db_path):
    con = sqlite3.connect(model_db_path, timeout=120)
    cur = con.cursor()
    try:
        cur.execute("insert into Signal values (?)", (1,))
        con.commit()
    except Exception as e:
        return 0
    finally:
        con.close()


def get_signal(model_db_path):
    con = sqlite3.connect(model_db_path, timeout=120)
    cur = con.cursor()
    try:
        rowid = cur.execute("select rowid from Signal order by rowid desc limit 1").fetchone()[0]
        return rowid
    except Exception as e:
        return 0
    finally:
        con.close()


def testing(args, locally=False):
    """
    Perform validation and testing
    Args:
        args: the training args
        locally: `locally` meaning not running on slurm with the training process
    """
    import torch
    import logging
    out_dir = args.outdir or args.model_db_path.replace("model.db", "")
    log_file = os.path.join(out_dir, 'testing.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    log = logging.getLogger(__name__)
    print(args.model_db_path)

    if not locally:
        # wait until trainer exits
        log.info("Waiting for trainer to exit")
        print("Waiting for trainer to exit")
        # time.sleep(60 * 5)
        while True:
            time.sleep(5)
            if get_signal(args.model_db_path) == args.world_size:
                log.info("Trainer exited. Starting testing")
                print("Trainer exited. Starting testing")
                break

    if args.submitit.log_dir is not None:
        # running in another node, needs to set start method
        torch.multiprocessing.set_start_method("spawn")

    split_rank, num_splits = 0, 1
    # out_dir = args.outdir or args.model_db_path.replace("model.db", "")
    max_steps = 100
    train_dataset_path = args.dataset.pydantic_dataset_path_dev  # avoid using large traing file
    test_dataset_path = args.dataset.pydantic_dataset_path_test
    model_db_path = args.model_db_path
    model_rowid = get_model_rowid(args.model_db_path)
    args_ = split_rank, num_splits, test_dataset_path, train_dataset_path, model_db_path, model_rowid, out_dir, max_steps, args.dataset.exclude_sets, args.sampling
    policy_rollout = test_model(args_)
    args_ = split_rank, num_splits, train_dataset_path, train_dataset_path, model_db_path, model_rowid, out_dir, max_steps, args.dataset.exclude_sets, args.sampling
    policy_rollout_val = test_model(args_)
    for max_steps in [13, 25, 45, 100]:
        log.info(f"Validation with {max_steps=}")
        logstr = policy_rollout_val.eval_metrics(max_steps=max_steps)
        log.info(logstr)
        log.info(f"Testing with {max_steps=}")
        logstr = policy_rollout.eval_metrics(max_steps=max_steps)
        log.info(logstr)
    val_oz = policy_rollout_val.eval_metrics(max_steps=45, return_oz_metric=True)
    test_oz = policy_rollout.eval_metrics(max_steps=45, return_oz_metric=True)
    with open("cg_paper_exp/all_results.txt", "at") as f:
        f.write(f"val_oz={val_oz:7.3%}, test_oz={test_oz:7.3%}, {out_dir=}\n")
    write_signal(args.model_db_path)


if __name__ == "__main__":
    all_args = []
    main()  # for local run or saving hydra multirun
    submit_main()  # for submitit

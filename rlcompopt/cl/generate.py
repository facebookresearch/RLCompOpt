
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import os
import pickle
import random
import sqlite3
import multiprocessing
from itertools import product
from multiprocessing import cpu_count
from time import sleep
from hydra.utils import get_original_cwd, to_absolute_path

import torch.multiprocessing as torch_mp
import common_utils
import compiler_gym
import gym
import humanize
import hydra
import numpy as np
import pandas as pd
import submitit
import tabulate
import torch
import typer
from submitit.core.utils import FailedJobError
from hydra.core.hydra_config import HydraConfig
from scipy import stats as stats

from rlcompopt.pipeline.lib.types import TrajectoryDataset
from rlcompopt.env_wrapper.merge_db import batch_merge
from rlcompopt.pipeline.lib.types import PolicyRolloutDataset
from .dataset_statistics import get_stat_packed_args
from .generate_utils import (
    get_benchmarks,
    load_benchmarks_from_json,
    run_agent_packed_args,
)
from .generate_utils_online import run_agent_packed_args4, run_model_packed_args, make_env_ready_to_use

log = logging.getLogger(__file__)


@hydra.main(config_path="./conf", config_name="generate.yaml")
def main(args):
    log.info(common_utils.print_info(args))
    log.info(common_utils.pretty_print_args(args))
    hconfig = HydraConfig.get()

    outd = os.getcwd()  # use absolute path
    print(f"output_dir={outd}")
    args.outdir = outd
    args.model_db_path = os.path.join(outd, "model.db")

    nproc = args.nproc or cpu_count()
    if args.submitit.log_dir is not None:
        nproc = args.submitit.cpus_per_task

    if args.dataset_name is not None:
        with gym.make("llvm-v0") as env:
            benchmarks = list(
                get_benchmarks(env, args.dataset_name, exclude=args.benchmark_exclude)
            )
    # load_benchmarks_from_json may overwrite dataset_name
    if args.load_benchmarks_from_json is not None:
        benchmarks = load_benchmarks_from_json(
            args.load_benchmarks_from_json, args.json_key
        )
    if args.traj_data is not None:
        # read benchmarks from Kevin's feather files
        df = pd.read_feather(args.traj_data)
        benchmarks = df.benchmark.unique().tolist()

    if args.num_benchmarks is not None:
        benchmarks = benchmarks[: args.num_benchmarks]
    if args.generate_v4:
        # benchmarks are randomly sampled
        benchmarks = []

    log.info(f"Using {len(benchmarks)} benchmarks: {benchmarks}")

    jobs = [
        (i, benchmark, seed, args)
        for i, (benchmark, seed) in enumerate(
            product(benchmarks, range(args.seed, args.seed + args.benchmark_repeat))
        )
    ]

    func = get_stat_packed_args if args.get_stat else run_agent_packed_args

    if jobs:
        random.shuffle(jobs)  # load balance

    # Debug:
    # func(jobs[0])
    # return
    if args.pydantic_datasource is not None:
        args.pydantic_datasource = to_absolute_path(args.pydantic_datasource)
    if args.pydantic_val_dataset_path is not None:
        args.pydantic_val_dataset_path = to_absolute_path(args.pydantic_val_dataset_path)
    if args.pydantic_test_dataset_path is not None:
        args.pydantic_test_dataset_path = to_absolute_path(args.pydantic_test_dataset_path)
    if args.vocab_db_path is not None:
        args.vocab_db_path = to_absolute_path(args.vocab_db_path)

    def run_generate_v4(args0):
        """
        Run a job locally in a node with multiprocessing
        """
        args, node_idx, total_nodes = args0

        out_dir = hconfig.runtime.output_dir
        import logging
        log_file = os.path.join(out_dir, 'generate.log')
        logging.basicConfig(filename=log_file, level=logging.INFO)
        log = logging.getLogger(__name__)

        if args.gpu is not None:
            # need to use spawn to enable cuda multiprocessing
            mp = torch_mp
            torch_mp.set_start_method("spawn")
        else:
            mp = multiprocessing

        make_env_ready_to_use(all_=True)
        # online generation as the model is trained
        m = mp.Manager()
        cg_jobs = []
        model_jobs = []
        n_cg_per_model = nproc // args.n_model_workers
        assert args.n_model_workers * n_cg_per_model == nproc
        test_result_queue = m.Queue() if (args.online_test_json is not None or args.pydantic_val_dataset_path is not None) else None
        for i in range(args.n_model_workers):
            queue0 = m.Queue() if not args.run_model_locally else None
            queue1 = [
                m.Queue() if not args.run_model_locally else None
                for _ in range(n_cg_per_model)
            ]
            cg_jobs.extend(
                [
                    (
                        i * n_cg_per_model + j + node_idx * nproc,
                        args.seed + i * n_cg_per_model + j,
                        args,
                        queue0,
                        queue1[j],
                        j,
                        i,
                        nproc * total_nodes,
                        test_result_queue,
                    )
                    for j in range(n_cg_per_model)
                ]
            )
            model_jobs.append((i, args, queue0, queue1, i))
        cg_pool = mp.Pool(nproc)
        print("Starting pool...")
        results = cg_pool.imap_unordered(run_agent_packed_args4, cg_jobs)
        print("Started pool")
        log.info(f"Started process pool for generating trajectories")
        if not args.run_model_locally and args.model_db_path is not None:
            model_pool = mp.Pool(args.n_model_workers)
            results2 = model_pool.imap_unordered(run_model_packed_args, model_jobs)
            log.info(f"Started process pool for model computation")

        # get testing results
        if test_result_queue is not None and args.pydantic_val_dataset_path is not None:
            assert total_nodes == 1  # can only handle one node

            def gather_results(rollouts):
                dataset = rollouts[0].benchmark_dataset
                policy_name = rollouts[0].policy_name
                samples = sum((roll.samples for roll in rollouts), [])
                output = PolicyRolloutDataset(
                    benchmark_dataset=dataset,
                    policy_name=policy_name,
                    samples=samples)
                metrics = output.eval_metrics(args.online_test_max_step)
                improve_over_oz = output.eval_metrics(args.online_test_max_step, return_oz_metric=True)
                return output, metrics, improve_over_oz

            test_result_template = {"test": {}, "validation": {}}
            test_result_batches = []
            printed = {"test": [], "validation": []}
            scores = {"test": [], "validation": []}
            metrics_list = {"test": [], "validation": []}
            stop_proc = False
            outdir = os.path.join(hconfig.runtime.output_dir, "eval")
            os.makedirs(outdir, exist_ok=True)
            while True:
                tag, model_rowid, policy_rollout, rank = test_result_queue.get()
                inserted = False 
                for test_res in test_result_batches:
                    if rank in test_res[tag].keys():
                        continue
                    test_res[tag][rank] = (model_rowid, policy_rollout)
                    inserted = True
                if not inserted:
                    test_res = copy.deepcopy(test_result_template)
                    test_result_batches.append(test_res)
                    test_res[tag][rank] = (model_rowid, policy_rollout)
                for j_, test_res in enumerate(test_result_batches):
                    for tg in ["test", "validation"]:
                        if len(test_res[tg].keys()) == nproc and j_ not in printed[tg]:
                            printed[tg].append(j_)
                            rollouts = list([vv[1] for vv in test_res[tg].values()])
                            model_rowids = list([vv[0] for vv in test_res[tg].values()])
                            output, metrics, imp_oz = gather_results(rollouts)
                            log.info(f"Testing split: idx={j_}, tag={tg}: improvement_over_oz={imp_oz}")
                            rowids = np.array(model_rowids)
                            mode_result = stats.mode(rowids)
                            log.info(f"{model_rowids=}, {mode_result}")
                            log.info(f"{metrics}")
                            filename = os.path.join(outdir, f"{tg}_{j_}.json")
                            log.info(f"Saving to {filename}")
                            output.save(filename)
                            scores[tg].append(imp_oz)
                            metrics_list[tg].append(metrics)
                            if tg == "validation" and len(scores[tg]) >= args.early_stop_patience:
                                max_score = max(scores[tg])
                                max_idx = scores[tg].index(max_score)
                                dist2max = len(scores[tg]) - 1 - max_idx
                                if dist2max >= args.early_stop_patience:
                                    # no improvement in last `early_stop_patience` validations
                                    stop_proc = True
                                    log.info("Early stopping...")
                                    log.info(f"Best metrics with {max_idx=}")
                                    log.info(metrics_list["validation"][max_idx])
                                    log.info(metrics_list["test"][max_idx])
                if stop_proc:
                    log.info("Terminating process pool...")
                    cg_pool.terminate()
                    cg_pool.join()
                    log.info("Finished task")
                    # signal end of task
                    filename = os.path.join(out_dir, f"term.signal")
                    with open(filename, "wb") as f:
                        pass
                    return

        if test_result_queue is not None and args.online_test_json is not None:
            while True:
                try:
                    connection = sqlite3.connect(args.model_db_path, timeout=3200)
                    cursor = connection.cursor()
                    break
                except Exception as e:
                    sleep(2)
                    print(f"Failed to connect to model database: {e}. Retrying...")
            while True:
                benchmark_metrics = {}
                for i in range(len(cg_jobs) * 2):
                    tag, model_rowid, r = test_result_queue.get()
                    metrics = benchmark_metrics.get(tag, None)
                    if metrics is None:
                        metrics = {}
                        benchmark_metrics[tag] = metrics
                    metrics.update(r)
                records = []
                for tag, metrics in benchmark_metrics.items():
                    print(f"[{tag}]")
                    total_metric, mean_metric, rows, table_str = get_metrics(metrics)
                    rows = pickle.dumps(rows)
                    records.append(
                        (model_rowid, tag, total_metric, mean_metric, rows, table_str)
                    )
                try:
                    cursor.executemany(
                        "INSERT INTO Performance VALUES (?, ?, ?, ?, ?, ?)", records
                    )
                    connection.commit()
                except Exception as e:
                    log.info(f"Failed to insert into Model Performance: {e}")

        # gather results
        benchmark_metrics = {}
        for r in results:
            benchmark_metrics.update(r)
        get_metrics(benchmark_metrics)

        if not args.run_model_locally and args.model_db_path is not None:
            for r in results2:
                pass
        return

    if args.submitit.log_dir is not None:
        args.n_model_workers = args.submitit.gpus_per_node

    if args.generate_v4:
        jobs_list = []
        timeout_min = 60
        time_prep_data = 10
        num_jobs = args.submitit.jobs_per_task or 1
        if args.traj_db is not None:
            raise RuntimeError("Should not go here.")
            # deprecated
        elif args.pydantic_datasource is not None and args.pydantic_val_dataset_path is None:
            raise RuntimeError("Should not go here.")

        elif args.pydantic_datasource is not None and args.pydantic_val_dataset_path is not None:
            # online generation
            jobs_list = [(args, 0, 1)]

            if args.submitit.log_dir is not None:
                executor = submitit.AutoExecutor(folder=args.submitit.log_dir)
                executor.update_parameters(
                    cpus_per_task=nproc,
                    timeout_min=args.submitit.timeout_min,
                    slurm_partition=args.submitit.partition,
                    mem_gb=args.submitit.mem_gb,
                    gpus_per_node=args.submitit.gpus_per_node,
                    tasks_per_node=1,
                    nodes=1,
                    slurm_constraint=args.submitit.constraint,
                    slurm_signal_delay_s=120,
                )
                executor.update_parameters(name="gen_online")
                slurm_jobs = executor.submit(run_generate_v4, jobs_list[0])
                print(f"Submitted job_id: {slurm_jobs.job_id}")
            else:
                run_generate_v4(jobs_list[0])

            return

        if args.submitit.log_dir is not None:
            assert num_jobs <= 80 and num_jobs * nproc < 5120  # cannot exceed limits
            timeout_min = max(timeout_min, 5) + time_prep_data
            assert timeout_min < 72 * 60
            print(f"{timeout_min=}, {num_jobs=}")
            executor = submitit.AutoExecutor(folder=args.submitit.log_dir)
            executor.update_parameters(
                cpus_per_task=nproc,
                timeout_min=timeout_min,
                slurm_partition=args.submitit.partition,
            )
            slurm_jobs = executor.map_array(run_generate_v4, jobs_list)
            for job in slurm_jobs:
                try:
                    print(job.results())
                except (FailedJobError, RuntimeError) as e:
                    print(f"Job {job.job_id} failed: {e}. Info: {job.get_info()}")
        else:
            for job in jobs_list:
                run_generate_v4(job)
        if args.pydantic_datasource is not None:
            # merge db
            print("Start to merge db")
            outfiles = os.listdir(outdir)
            first = os.path.join(outdir, outfiles[0])
            print(f"First db: {first}")
            print(f"Move first db to {dst}")
            os.rename(first, dst)
            print(f"Merge db in {outdir} to {dst}")
            if len(outfiles) > 1:
                batch_merge(outdir, dst)
        return

    # other generation/evaluation logic
    # wrap the multiprocessing code into a function so that we can use submitit
    def run_jobs(jobs_):

        out_dir = hconfig.runtime.output_dir
        import logging
        log_file = os.path.join(out_dir, 'generate.log')
        logging.basicConfig(filename=log_file, level=logging.INFO)
        log = logging.getLogger(__name__)

        if args.gpu is not None:
            # need to use spawn to enable cuda multiprocessing
            mp = torch_mp
            torch_mp.set_start_method("spawn")
        else:
            mp = multiprocessing

        total_step_count = 0
        best_returns = []
        all_return_history = []

        with mp.Pool(nproc) as pool:
            results = pool.imap_unordered(func, jobs_)
            with typer.progressbar(results, length=len(jobs_)) as progress:
                if args.get_stat:
                    for result in progress:
                        log.info(result)
                    return

                total_ir_oz = 0
                covered_rate_all = 0
                total_count = 0
                raw_best_returns = []
                benchmark_raw_best_returns = {}
                benchmark_oz = {}
                benchmark_best_returns = {}
                for (
                    benchmark,
                    seed,
                    step_count,
                    best_return,
                    IrInstructionCountOz,
                    best_actions,
                    best_cmdline,
                    return_history,
                    covered_rate,
                ) in progress:
                    total_step_count += step_count
                    current_result = dict(
                        benchmark=benchmark,
                        seed=seed,
                        step_count=step_count,
                        best_return=best_return / IrInstructionCountOz,
                        raw_best_return=int(best_return),
                        Ir_count_Oz=IrInstructionCountOz,
                        covered_rate=covered_rate,
                        best_actions=best_actions,
                        best_cmdline=best_cmdline,
                        history=return_history,
                    )
                    total_ir_oz += IrInstructionCountOz
                    covered_rate_all += covered_rate
                    total_count += 1
                    log.info(f"json_str_one: {json.dumps(current_result)}")
                    best_returns.append(best_return / IrInstructionCountOz)
                    raw_best_returns.append(best_return)
                    all_return_history.append(current_result)
                    benchmark = benchmark.replace("benchmark://", "").split("-")[0]
                    if benchmark not in benchmark_best_returns.keys():
                        benchmark_best_returns[benchmark] = [
                            best_return / IrInstructionCountOz
                        ]
                        benchmark_raw_best_returns[benchmark] = [best_return]
                        benchmark_oz[benchmark] = [IrInstructionCountOz]
                    else:
                        benchmark_best_returns[benchmark].append(
                            best_return / IrInstructionCountOz
                        )
                        benchmark_raw_best_returns[benchmark].append(best_return)
                        benchmark_oz[benchmark].append(IrInstructionCountOz)
        log.info(f"Processed {humanize.intcomma(total_step_count)} steps")

        best_returns = torch.FloatTensor(np.array(best_returns))  # n x 1
        total_percent_oz = sum(raw_best_returns) / total_ir_oz
        log.info(f"Summary for {args.model_path}")
        log.info(f"Covered_rate: {covered_rate_all / total_count}")
        rows = []
        total_i = 0
        mean_i = 0
        for bm, val in benchmark_best_returns.items():
            dataset_total_improvement = sum(benchmark_raw_best_returns[bm]) / sum(
                benchmark_oz[bm]
            )
            dataset_mean_improvement = sum(val) / len(val)
            rows.append(
                [
                    bm,
                    f"{dataset_total_improvement: 6.1%}",
                    f"{dataset_mean_improvement: 6.1%}",
                ]
            )
            total_i += dataset_total_improvement
            mean_i += dataset_mean_improvement
        n_dataset = len(benchmark_best_returns)
        rows.append(
            ["[mean]", f"{total_i / n_dataset: 6.1%}", f"{mean_i / n_dataset: 6.1%}"]
        )
        table = tabulate.tabulate(
            rows, headers=["Dataset", "Total", "Mean"]
        )  # , tablefmt='latex_booktabs')
        log.info(f"{table}")
        log.info(f"Total percent improved over Oz: {total_percent_oz}")
        log.info(
            f"Avg percent improved over Oz: {best_returns.mean().tolist()} ± {best_returns.std().tolist()} [{best_returns.size(0)}]"
        )

        # dump all return history with json
        log.info(f"json_str: {json.dumps(all_return_history)}")

    if args.submitit.log_dir is not None:
        jobs_per_task = args.submitit.jobs_per_task
        jobs_list = [
            jobs[i : i + jobs_per_task] for i in range(0, len(jobs), jobs_per_task)
        ]
        executor = submitit.AutoExecutor(folder=args.submitit.log_dir)
        executor.update_parameters(
            cpus_per_task=args.submitit.cpus_per_task,
            timeout_min=args.submitit.timeout_min,
            slurm_partition=args.submitit.partition,
        )
        slurm_jobs = executor.map_array(run_jobs, jobs_list)
        for job in slurm_jobs:
            try:
                print(job.results())
            except (FailedJobError, RuntimeError) as e:
                print(f"Job {job.job_id} failed: {e}. Info: {job.get_info()}")
        return
    else:
        run_jobs(jobs)


def get_metrics(benchmark_metrics):
    # summarize the metric results
    # total/averaged percent improvement over Oz
    benchmark_raw_best_returns = {}
    benchmark_oz = {}
    benchmark_best_returns = {}
    for bm, (_, best_return, IrInstructionCountOz) in benchmark_metrics.items():
        best_return = float(best_return)
        benchmark = bm.replace("benchmark://", "").split("-")[0]
        if benchmark not in benchmark_best_returns.keys():
            benchmark_best_returns[benchmark] = [best_return / IrInstructionCountOz]
            benchmark_raw_best_returns[benchmark] = [best_return]
            benchmark_oz[benchmark] = [IrInstructionCountOz]
        else:
            benchmark_best_returns[benchmark].append(best_return / IrInstructionCountOz)
            benchmark_raw_best_returns[benchmark].append(best_return)
            benchmark_oz[benchmark].append(IrInstructionCountOz)
    # compute results for each dataset
    rows = []
    total_i = 0
    mean_i = 0
    for bm, val in benchmark_best_returns.items():
        dataset_total_improvement = sum(benchmark_raw_best_returns[bm]) / sum(
            benchmark_oz[bm]
        )
        dataset_mean_improvement = sum(val) / len(val)
        rows.append(
            [
                bm,
                f"{dataset_total_improvement: 6.1%}",
                f"{dataset_mean_improvement: 6.1%}",
            ]
        )
        total_i += dataset_total_improvement
        mean_i += dataset_mean_improvement
    n_dataset = len(benchmark_best_returns)
    total_metric = total_i / n_dataset
    mean_metric = mean_i / n_dataset
    rows.append(["(mean)", f"{total_metric: 6.1%}", f"{mean_metric: 6.1%}"])
    table = tabulate.tabulate(rows, headers=["Dataset", "Total", "Mean"])
    print(table)
    log.info(f"{table}")
    return total_metric, mean_metric, rows, f"{table}"


if __name__ == "__main__":
    main()

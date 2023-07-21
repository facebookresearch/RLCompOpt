
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import json
import os
import pickle
import sqlite3
import time
import logging
from typing import List

import networkx as nx
from depq import DEPQ

import compiler_gym  # noqa
import numpy as np
import submitit
import torch
from torch.distributions.categorical import Categorical
from torch.multiprocessing import Process

from rlcompopt.cl.generate_utils_online import GPUworker
from rlcompopt.env_wrapper.parsing_utils import FeatureExtractor
from rlcompopt.env_wrapper.pyg_utils import dgl2pyg
from rlcompopt.pipeline.lib.types import TrajectoryDataset, BenchmarkDataset, PolicyRolloutSample, PolicyRolloutDataset

logger = logging.getLogger(__name__)

def write(string, file):
    with open(file, "w") as f:
        f.write(string)


def load_datasets_from_json(json_file, key, min_ir, max_ir):
    with open(json_file, "r") as f:
        data = json.load(f)
    benchmarks = {}
    for benchmark, v in data.items():
        bm = v[key]  # a list of [ir_count, benchmark_name]
        filtered_bm = [b for i, b in bm if min_ir <= i <= max_ir]
        if filtered_bm:
            assert isinstance(bm[0][0], int) and isinstance(bm[0][1], str)
            # only use non-empty ones
            benchmarks[benchmark] = filtered_bm
    return benchmarks


def create_model(model_db_path, model_rowid, model_rank, device="cuda"):
    return GPUworker(
        None,
        [None],
        model_rank,
        device=device,
        model_db_path=model_db_path,
        model_capacity=1e8,
        load_full_rate=0,
        job_full_rate=0,
        wait_time=0,
        load_model_frequency=1e20,
        run_locally=True,
        model_rowid=model_rowid,
        eval_model=True,
    )


def get_feat_extractor(vocab_db_path):
    feature_extractor = FeatureExtractor(
        vocab_db_path,
        online_update=False,
        graph_version=1,
    )
    return feature_extractor


class Environment:
    def __init__(
        self,
        model_db_path,
        model_rowid,
        model_rank,
        vocab_db_path,
        max_step=45,
        benchmarks=None,
        train_dataset_path=None,
        out_path=None,
        seed=0,
        sampling=True,
    ):
        print("init")
        self.env = compiler_gym.make(
            "llvm-autophase-ic-v0",
            reward_space="IrInstructionCount",
        )
        print("done init")
        self.max_step = max_step
        self.seed = seed
        self.num_actions = self.env.action_space.n
        self.benchmark_metrics = {}
        self.benchmarks = benchmarks
        print("loading")
        dataset = TrajectoryDataset.load(train_dataset_path)
        print("loaded data")
        self.actionseqs = [seq.actions for seq in dataset.action_sequences.actionseqs]
        self.iter = 0
        self.rank = model_rank
        self.model = create_model(model_db_path, model_rowid, model_rank)
        self.use_autophase = self.model.use_autophase
        self.feature_extractor = get_feat_extractor(vocab_db_path)
        self.features = ["IrInstructionCount"]
        self.out_path = out_path
        self.sampling = sampling

    def run(self):
        samples = []
        for benchmark in self.benchmarks:
            result = self.run_a_benchmark(benchmark, self.sampling)
            if result is not None:
                samples.append(result)
        with open(self.out_path, "wb") as f:
            pickle.dump(samples, f)

    def run_a_benchmark(self, benchmark, sampling=True):
        raw_obs = self.reset(benchmark)
        top_seqs = self.get_model_action(raw_obs, return_prob=sampling)
        if sampling:
            sampled = set()
            repeated_cnt = 0
        try:
            steps = 0
            all_counts = [self.orig_size]
            # for idx in top_seqs:
            i = 0
            while True:
                if sampling:
                    while True:
                        assert len(sampled) < len(self.actionseqs)
                        idx = top_seqs.sample().item()
                        if idx not in sampled:
                            sampled.add(idx)
                            break
                        else:
                            repeated_cnt += 1
                            if repeated_cnt > 10:
                                repeated_cnt = 0
                                # increase the temperature to enable sampling from more choises
                                self.logit = self.logit * 0.5
                                top_seqs = Categorical(logits=self.logit)
                else:
                    idx = top_seqs[i]
                    i += 1
                actions = self.actionseqs[idx]
                ir_counts = self.run_an_episode(benchmark, actions)
                all_counts.extend(ir_counts)
                steps += len(actions)
                if steps >= self.max_step:
                    break
            traj = PolicyRolloutSample(
                benchmark=benchmark, ir_original=self.orig_size, 
                ir_compiler=self.compiler_size, ir_search_trajectory=all_counts)
            return traj
        except Exception as e:
            print(f"Fail on {benchmark}: {e}")

    def run_an_episode(self, benchmark, actions=None):
        self.env.reset(benchmark=benchmark)
        # ir0 = self.env.observation["IrInstructionCountO0"]
        self._reset_obs()

        ir_counts = []
        for a in actions:
            ir_count = self.step(a)
            ir_counts.append(ir_count)
        return ir_counts

    @staticmethod
    def to_str(seq):
        return " ".join(str(s) for s in seq)

    @staticmethod
    def str2list(s):
        """
        Convert a string like 'a b c' to a list of integers like [a, b, c]
        """
        if len(s) == 0:
            s = []
        else:
            s = [int(a) for a in s.split(" ")]
        return s

    def step(self, action: int):
        states, reward, done, meta = self.env.step(
            action=action,
            observation_spaces=self._observation_spaces,
        )
        if done:
            raise RuntimeError("Trajectory ended with done.")
        raw_state, = states
        return raw_state

    def reset(self, benchmark):
        self.benchmark = benchmark
        seed = self.seed + str2int(benchmark)
        np.random.seed(seed)
        self.env.reset(benchmark=benchmark)
        self.orig_size = self.env.observation["IrInstructionCountO0"]
        self.compiler_size = self.env.observation["IrInstructionCountOz"]
        raw_state = self.orig_size
        # if self.compiler_size == 0:
        #     self.compiler_size = -1
        self.ir_obs = [raw_state]
        self.min_ir = raw_state
        self.best_action_seq = []
        self.best_ir_seq = [raw_state]
        self.iter = 0
        self._reset_obs()
        if self.use_autophase:
            raw_obs = self.env.observation["Autophase"]
        else:
            raw_obs = self.env.observation["Programl"]
        return raw_obs

    def _reset_obs(self):
        self._observation_spaces = [
            self.env.observation.spaces[feat] for feat in self.features
        ]

    def convert_graph(self, raw_graph):
        dgl_graph = self.feature_extractor.process_nx_graph(raw_graph)
        graph = dgl2pyg(dgl_graph)
        return graph

    def get_model_action(self, raw_obs, return_prob=False):
        if self.use_autophase:
            graph = None
            autophase = self._autophase2vec(raw_obs)
        else:
            graph = self.convert_graph(raw_obs)
            autophase = None

        action_history, reward_history = self.get_history_params()
        job = (self.rank, graph, autophase, action_history, reward_history)
        logit, value, repr_ = self.model.get_model_output_local(job)
        self.logit = logit.flatten()

        if return_prob:
            m = Categorical(logits=logit.flatten())
            return m
        # action = m.sample().item()
        top_actions = torch.sort(logit.flatten(), dim=0, descending=True)[1].cpu().tolist()
        return top_actions

    def get_history_params(self):
        action_history, reward_history = None, None
        return action_history, reward_history

    def _autophase2vec(self, raw_state):
        if isinstance(raw_state, str):
            raw_state = [int(x) for x in raw_state.split(" ")]
        autophase = torch.tensor(raw_state, dtype=torch.float)
        # if self.normalize_autophase:
        # normalize autophase instruction counts
        total_instructions = raw_state[51]
        autophase = autophase / total_instructions
        return autophase

def str2int(s):
    return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (2**31)


def run_opt(args):
    env = Environment(*args)
    env.run()


def make_env_ready_to_use(all_=False):
    """
    Make sure necessary files are downloaded and ready to use
    """
    with compiler_gym.make("llvm-v0") as env:
        env.reset(benchmark="benchmark://anghabench-v1/linux/drivers/gpu/drm/amd/display/dc/dcn10/extr_dcn10_optc.c_optc1_program_timing")
        if all_:
            env.reset(benchmark="benchmark://poj104-v1/47/4049")
            env.reset(benchmark="benchmark://opencv-v0/204")
            env.reset(benchmark="benchmark://github-v0/9444")
            env.reset(benchmark="benchmark://linux-v0/46")
            env.reset(benchmark="generator://llvm-stress-v0/808")
            env.reset(benchmark="benchmark://clgen-v0/449531a637c8f33bdf11263fe6b9c5ea657be851")
            env.reset(benchmark="benchmark://tensorflow-v0/490")
            env.reset(benchmark="generator://csmith-v0/942")
            env.reset(benchmark="benchmark://blas-v0/6")
            env.reset(benchmark="benchmark://chstone-v0/gsm")
            env.reset(benchmark="benchmark://mibench-v1/bitcount-1")
            env.reset(benchmark="benchmark://npb-v0/17")


def test_model(args=None):
    split_rank, num_splits, test_dataset_path, train_dataset_path, model_db_path, model_rowid, out_dir, max_steps, excludes, sampling = args
    # filename = 'val_result.json' if 'trajdataset' in test_dataset_path else 'test_result.json'
    filename = 'val_result.json' if 'trajdataset' in test_dataset_path else 'test_result.json'
    save_path = os.path.join(out_dir, filename)
    print(f"{save_path=}")
    print(f"Testing with model {model_db_path} at row {model_rowid}")
    if os.path.exists(save_path):
        output = PolicyRolloutDataset.load(save_path)
        return output
    print(os.environ["PATH"])
    print(os.environ["COMPILER_GYM_SITE_DATA"])
    make_env_ready_to_use(all_=True)
    print(f"{split_rank=}, {num_splits=}")
    if 'trajdataset' in test_dataset_path:
        dataset = TrajectoryDataset.load(test_dataset_path).benchmark_dataset
    else:
        dataset = BenchmarkDataset.load(test_dataset_path)
    benchmarks = dataset.benchmarks

    if excludes is not None:
        excludes = excludes.split("_")
        def is_excluded(b: str):
            for dataset in excludes:
                if b.find(dataset) >= 0:
                    return True
            return False
        benchmarks = [bb for bb in benchmarks if not is_excluded(bb)]

    benchmarks = benchmarks[split_rank::num_splits]
    print(f"size of dataset: {len(benchmarks)}")
    cpu_count = int(os.environ["NUM_CPU"])  # os.cpu_count()
    print(f"cpu_count={cpu_count}")
    seed = 0
    vocab_db_path = "data/all_ssl_vocab.db"
    n_gpus = int(os.environ["NUM_GPU"])
    out_path = [os.path.join(out_dir, f"split_rank{split_rank}_{i}.pkl") for i in range(cpu_count)]
    jobs = [
        (
            model_db_path,
            model_rowid,
            i % n_gpus,
            vocab_db_path,
            max_steps,
            benchmarks[i::cpu_count],
            train_dataset_path,
            out_path[i],
            seed,
            sampling,
        )
        for i in range(cpu_count)
    ]

    os.makedirs(out_dir, exist_ok=True)

    pros = []
    # run_opt(jobs[0])
    # return
    t0 = time.time()
    for job in jobs:
        p = Process(target=run_opt, args=(job,))
        p.start()
        print("started p")
        pros.append(p)
    for p in pros:
        p.join()
    print("Used time:", time.time() - t0)

    # merge
    all_samples = []
    for path in out_path:
        with open(path, "rb") as f:
            samples = pickle.load(f)
            all_samples.extend(samples)
    output = PolicyRolloutDataset(
        benchmark_dataset=dataset,
        policy_name='policy0',
        samples=all_samples)

    output.save(save_path)
    metrics = output.get_metric(max_steps=45)
    metrics["_all_"] = sum(metrics.values()) / len(metrics)
    print(metrics)
    logger.info(str(metrics))
    return output


def check_output(path):
    output = PolicyRolloutDataset.load(path)
    metrics = output.get_metric(max_steps=45)
    metrics["_all_"] = sum(metrics.values()) / len(metrics)
    print(metrics)
    import ipdb
    ipdb.set_trace()
    x = 0


def get_model_rowid(model_db_path):
    con = sqlite3.connect(model_db_path)
    cur = con.cursor()
    try:
        rowid = cur.execute("select model_id from ValBest order by rowid desc limit 1").fetchone()[0]
    except Exception as e:
        print(f"Cannot read model_id from ValBest: {e}")
        rowid = cur.execute("select rowid from Models order by rowid desc limit 1").fetchone()[0]
    return rowid


def testing(args_):
    out_dir, pydantic_dataset_path_dev, pydantic_dataset_path_test, exclude_sets, sampling = args_
    import torch
    import logging
    model_db_path = os.path.join(out_dir, "model.db")
    log_file = os.path.join(out_dir, 'testing.log')
    logging.basicConfig(filename=log_file, level=logging.INFO)
    log = logging.getLogger(__name__)
    log.info("Starting new eval ..........")
    torch.multiprocessing.set_start_method("spawn")

    split_rank, num_splits = 0, 1
    # out_dir = args.outdir or args.model_db_path.replace("model.db", "")
    max_steps = 100
    train_dataset_path = pydantic_dataset_path_dev  # avoid using large traing file
    test_dataset_path = pydantic_dataset_path_test
    model_rowid = get_model_rowid(model_db_path)
    args_ = split_rank, num_splits, test_dataset_path, train_dataset_path, model_db_path, model_rowid, out_dir, max_steps, exclude_sets, sampling
    policy_rollout = test_model(args_)
    args_ = split_rank, num_splits, train_dataset_path, train_dataset_path, model_db_path, model_rowid, out_dir, max_steps, exclude_sets, sampling
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

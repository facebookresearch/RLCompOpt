
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pickle
import random
import sqlite3
from datetime import datetime
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from threading import Thread
from time import sleep, time
from typing import Callable, List, Mapping, Tuple

import compiler_gym
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from compiler_gym.util.timer import Timer
from torch.distributions.categorical import Categorical
from torch_geometric.data import Batch, Data

from rlcompopt.cl.data_socket import DataServer
from rlcompopt.cl.generate_utils import (
    load_benchmarks_from_json,
    load_datasets_from_json,
)
from rlcompopt.env_wrapper.wrapper_online import (
    CompilerOnPolicyWrapper,
    CompilerPPOWrapper,
)
from rlcompopt.pipeline.lib.types import (
    TrajectoryDataset, 
    BenchmarkDataset, 
    PolicyRolloutSample, 
    PolicyRolloutDataset
)

# NOTE: logging is not working with multiprocessing spawn
# log = logging.getLogger(__file__)

def write(string: str, file="./log.txt"):
    with open(file, "at") as f:
        f.write(string)


class GPUworker(Thread):
    def __init__(
        self,
        queue0: Queue,  # the common quque to receive jobs from the agents
        queue1: List[Queue],  # the list of queues to send the results to agents
        local_rank: int,
        device: str,
        model_db_path: str,
        model_capacity: int,  # the max amount of data the model can consume; for graphs, it is #nodes; for autophase, it is batch size
        load_full_rate: float,  # trigger computation if current_load reaches model_capacity * load_full_rate
        job_full_rate: float,  # trigger computation if #jobs reaches num_clients * job_full_rate
        wait_time: float,  # wait time in seconds before trigger computation
        load_model_frequency: float,  # how long to wait to load next model
        run_locally=False,  # whether to use the model in the local process
        model_rowid=None,
        eval_model=False,
    ):
        super().__init__()
        self.model_db_path = model_db_path
        self.device = f"{device}:{local_rank}" if device == "cuda" else "cpu"

        self.queue_recv = queue0
        self.queues_send = queue1
        self.num_clients = len(queue1)
        self.model_capacity = model_capacity
        self.job_full_rate = job_full_rate
        self.load_full_rate = load_full_rate
        self.wait_time = wait_time
        self.load_model_frequency = load_model_frequency
        self.run_locally = run_locally
        self.model_rowid = model_rowid
        self.eval_model = eval_model

        self.job_buffer = []
        self.current_load = 0
        self.total_job = 0
        self.total_computation_time = 0
        self.last_load_time = time()

        if self.run_locally:
            self.init_model()

    def init_model(self):
        self.connection = sqlite3.connect(self.model_db_path, timeout=3200)
        self.cursor = self.connection.cursor()
        rec = []
        while not rec:
            try:
                if self.model_rowid is not None:
                    rec = list(
                        self.cursor.execute(
                            f"SELECT rowid, * FROM Models where rowid = {self.model_rowid}"
                        )
                    )
                else:
                    rec = list(
                        self.cursor.execute(
                            "SELECT rowid, * FROM Models ORDER BY rowid DESC LIMIT 1"
                        )
                    )
            except sqlite3.OperationalError:
                # database not yet ready
                pass
            finally:
                sleep(1)
        rowid, config, kwargs, state_dict, state_dict_ema = rec[0]
        config = pickle.loads(config)
        kwargs = pickle.loads(kwargs)
        state_dict = pickle.loads(state_dict)
        state_dict_ema = pickle.loads(state_dict_ema)
        if state_dict_ema is not None:
            state_dict = state_dict_ema
        model: torch.nn.Module = hydra.utils.instantiate(config, **kwargs)
        msg = model.load_state_dict(state_dict)
        if state_dict_ema is not None:
            assert model.use_cl
        print(f"Initialized model with the checkpoint from database row {rowid}: {msg}")
        self.last_idx = rowid
        self.model = model.to(self.device).eval()
        self.use_autophase = model.use_autophase
        self.use_history = model.use_history
        self.use_value_function = model.use_value_function
        self.use_ppo = model.use_ppo
        self.use_reinforce = model.use_reinforce
        self.use_action_histogram = model.use_action_histogram
        self.num_actions = model.num_actions
        self.action_histogram_steps = model.action_histogram_steps

    def load_model(self):
        if self.eval_model:
            # not to load models as in evaluation
            return
        rec = list(
            self.cursor.execute(
                f"SELECT rowid, state_dict, ema_state_dict FROM Models where rowid > {self.last_idx} ORDER BY rowid DESC LIMIT 1"
            )
        )
        if rec:
            rowid, state_dict, state_dict_ema = rec[0]
            state_dict: Mapping[str, torch.Tensor] = pickle.loads(state_dict)
            state_dict_ema: Mapping[str, torch.Tensor] = pickle.loads(state_dict_ema)
            if state_dict_ema is not None:
                state_dict = state_dict_ema
            for k, v in state_dict.items():
                state_dict[k] = v.to(device=self.device)
            self.model.load_state_dict(state_dict)
            self.last_idx = rec[0][0]

    def get_model_output(self):
        samples = self.job_buffer
        if not samples:
            return
        self.total_job += len(samples)
        self.job_buffer = []
        self.current_load = 0
        with Timer() as timer:
            samples = list(zip(*samples))

            if samples[1][0] is not None:
                graphs = Batch.from_data_list(samples[1]).to(self.device)
                autophase = None
            else:
                graphs = None
                autophase = torch.stack(samples[2]).to(self.device)
            if self.use_history:
                actions, rewards = samples[3], samples[4]
                seq_pos = torch.tensor(
                    [a.shape[0] for a in actions], dtype=torch.long, device=self.device
                )
                max_seq_len = max(a.shape[0] for a in actions)
                if self.use_action_histogram:
                    histogram = torch.zeros(len(actions), self.num_actions, dtype=torch.float)
                    for ii, act_history in enumerate(actions):
                        for a in act_history.tolist():
                            histogram[ii, a] += 1
                    action_history = histogram.to(self.device, non_blocking=True) / self.action_histogram_steps
                else:
                    action_history = torch.stack(
                        [
                            F.pad(a, (0, max_seq_len - a.shape[0]), "constant", 0)
                            for a in actions
                        ]
                    ).to(self.device, non_blocking=True)
                reward_history = torch.stack(
                    [
                        F.pad(r, (0, max_seq_len - r.shape[0]), "constant", 0)
                        for r in rewards
                    ]
                ).to(self.device, non_blocking=True)
                a_r = action_history, reward_history
                padding_mask = torch.ones(
                    len(actions), max_seq_len, dtype=torch.bool, device=self.device
                )
                for i in range(len(actions)):
                    padding_mask[i, : actions[i].shape[0]] = 0

                reward_sign = torch.sign(reward_history).to(torch.long) + 2
                start = torch.zeros(
                    reward_sign.shape[0], 1, dtype=torch.long, device=self.device
                )
                reward_sign_hist = torch.cat([start, reward_sign], dim=1)
            else:
                a_r, padding_mask, reward_sign_hist, seq_pos = (None,) * 4

            with torch.no_grad():
                if self.use_ppo or self.use_reinforce:
                    logit, other, repr_s = self.model.get_logits(
                        graphs,
                        autophase=autophase,
                        action_rewards=a_r,
                        padding_mask=padding_mask,
                        reward_sign_hist=reward_sign_hist,
                        seq_pos=seq_pos,
                    )
                else:
                    logit, other = self.model.get_logits(graphs, autophase=autophase)
                    repr_s = None

                if repr_s is not None:
                    repr_s = repr_s.cpu()
                else:
                    repr_s = (None,) * logit.shape[0]
                output = logit.cpu()
                if self.use_value_function:
                    values = other.cpu()
                else:
                    values = (None,) * output.shape[0]

            # send the results to the clients
            for client_id, rec, val, repr_ in zip(samples[0], output, values, repr_s):
                if self.run_locally:
                    return rec, val, repr_
                self.queues_send[client_id].put_nowait((rec, val, repr_))

        self.total_computation_time += timer.time
        if self.total_job >= 200:
            half_time = (
                self.total_computation_time / self.total_job * self.num_clients * 0.5
            )
            if self.wait_time > half_time:
                self.wait_time = half_time

    def _job_size(self, job: Tuple[int, Data, torch.Tensor]):
        if self.use_autophase:
            return 1
        return job[1].num_nodes

    def get_model_output_local(self, job):
        self.add_job(job)
        rec = self.get_model_output()
        if time() - self.last_load_time > self.load_model_frequency:
            self.load_model()
            self.last_load_time = time()
        return rec

    def run(self):
        self.init_model()  # can only create and use the sql connection in the same thread

        while True:
            try:
                job = self.queue_recv.get(timeout=self.wait_time)
            except Empty:
                job = None
            # try:
            if job is None:
                # have waited too long and got nothing, just do the computation
                self.get_model_output()
            else:
                self.add_job(job)
            if time() - self.last_load_time > self.load_model_frequency:
                self.load_model()
                self.last_load_time = time()
            # except Exception as e:
            #     log.info(f"Failed in model computation at rank {self.device}: {e}")

    def add_job(self, job):
        sz = self._job_size(job)
        if sz > self.model_capacity:
            if self.run_locally:
                return
            self.queues_send[job[0]].put_nowait(None)
            return
        self.job_buffer.append(job)
        self.current_load += sz
        if not self.run_locally and (
            len(self.job_buffer) >= self.num_clients * self.job_full_rate
            or self.current_load >= self.model_capacity * self.load_full_rate
        ):
            self.get_model_output()


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


class Benchmarks:
    def __init__(
        self,
        json_file,
        json_key="train",
        min_ir=0,
        max_ir=None,
        split_rank=None,
        num_splits=1,
        use_only_anghabench=False,
        num_benchmarks=None,
    ) -> None:
        self.json_file = json_file
        self.json_key = json_key
        self.min_ir = min_ir
        if max_ir is None:
            max_ir = float("inf")
        self.max_ir = max_ir
        if split_rank is None:
            self._datasets = load_datasets_from_json(
                self.json_file,
                key=self.json_key,
                min_ir=self.min_ir,
                max_ir=self.max_ir,
                num_benchmarks=num_benchmarks,
            )
            if use_only_anghabench:
                for k, v in self._datasets.items():
                    if k.lower().find("anghabench") >= 0:
                        self._datasets = {k: v}
                        break
            self._dataset_names = list(self._datasets.keys())
            print("Dataset size:", {k: len(v) for k, v in self._datasets.items()})
        else:
            self._benchmarks = load_benchmarks_from_json(
                self.json_file, key=self.json_key
            )

        self.split_rank = split_rank
        self.num_splits = num_splits

    def sample(self) -> str:
        if self.split_rank is None:
            # need to use np.random.choice for per-process specific randomness
            d_idx = np.random.choice(len(self._dataset_names))
            dataset = self._datasets[self._dataset_names[d_idx]]
            b_idx = np.random.choice(len(dataset))
            return dataset[b_idx]
        # for offline testing, split the benchmarks into disjoint sets and each worker deal with one subset
        # TODO: remove this part, should use __iter__
        if self.split_rank < len(self._benchmarks):
            bm = self._benchmarks[self.split_rank]
            self.split_rank += self.num_splits
        else:
            bm = None
        return bm

    def __iter__(self):
        return iter(self._benchmarks[self.split_rank :: self.num_splits])


class AgentWorker(Thread):
    """Worker thread to run a repeating agent.
    To stop the agent, set the :code:`alive` attribute of this thread to False.
    """

    def __init__(
        self,
        job_id: int,
        seed: int,
        args: argparse.Namespace,
        queue0: Queue,  # the common quque that all agents using the GPU can put something into, should put (local_rank, ...)
        queue1: Queue,  # the queue associated with the local rank from which the agent gets the result
        local_rank: int,  # the local index under a GPU
        make_env: Callable[[], CompilerOnPolicyWrapper],
        use_autophase=False,
        use_history=False,
        run_locally=False,
        create_model: Callable[[], GPUworker] = None,
        use_ppo=False,
        benchmark_split_rank=None,
        num_splits=1,
        test_split_rank=0,
        test_result_queue: Queue = None,  # a queue via which the workers can send the test results to the main process
        make_test_env: Callable[[], CompilerOnPolicyWrapper] = None,
        max_episodes=None,
        traj_db=None,
        pydantic_file=None,
        pydantic_val_dataset_path=None,  # for early stop in online RL training
        pydantic_test_dataset_path=None,
        simple_generation=False,  # dummy generation for getting the initial observations
    ):
        super().__init__()
        self._make_env = make_env

        # Send random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(
            seed
        )  # this will affect the seed of the random module of the whole process group
        np.random.seed(seed)
        self.seed = seed

        self.total_episode_count = 0

        self.reward_spaces = ["IrInstructionCount"]

        self.use_autophase = use_autophase
        self.features = ["IrSha1"]
        # self.features = ["IrSha1", "Programl", "Autophase", "IrInstructionCount"]
        self.use_history = use_history
        self.run_locally = run_locally
        self.create_model = create_model
        self.use_ppo = use_ppo
        self.use_random_action = args.model_db_path is None

        self.autophase_bounds = [
            [0, 1, 3, 7, 14, 28],
            [0, 1, 3, 6, 10, 18, 39, 59, 152],
            [0, 11, 16, 27, 45, 81, 133, 200, 295, 562],
            [0, 7, 12, 19, 31, 57, 94, 147, 230, 485],
            [0, 2, 4, 8, 16, 25, 41, 58, 106, 189],
            [0, 9, 14, 21, 33, 60, 101, 159, 243, 464],
            [0, 4, 7, 10, 19, 36, 55, 80, 122, 331],
            [0, 2, 3, 5, 10, 17, 32, 63],
            [0, 2, 4, 7, 14, 24, 45, 61, 109],
            [0, 6, 9, 16, 31, 49, 94, 123, 214, 430],
            [0, 1, 3, 5, 9, 13, 19, 32, 95],
            [0, 1, 3, 6, 9, 16, 39, 58, 113],
            [0, 1, 3, 7, 32],
            [1, 14, 21, 32, 54, 94, 156, 252, 377, 694],
            [0, 1, 3, 6, 10, 18, 42, 62, 110],
            [0, 15, 22, 34, 61, 104, 179, 265, 375, 713],
            [0, 1, 4, 6, 10, 16, 29, 68],
            [0, 1, 4, 11, 26, 65, 119, 374],
            [0, 22, 32, 49, 95, 156, 274, 389, 569, 1104],
            [1, 21, 32, 50, 100, 196, 279, 488, 806, 1666],
            [0, 7, 15, 25, 43, 81, 152, 275, 488, 1064],
            [0, 11, 19, 30, 55, 101, 161, 345, 661, 1749],
            [0, 8, 15, 24, 54, 93, 157, 223, 359, 790],
            [0, 9, 14, 21, 33, 60, 101, 159, 243, 464],
            [0, 3, 6, 11, 22, 46, 75, 128, 190, 340],
            [0, 1, 4, 16],
            [0, 1, 4, 7, 13, 24, 46, 75, 158, 446],
            [0, 1, 2, 4, 9, 20, 61, 128, 478],
            [0, 1, 3, 8, 18, 32, 67, 141],
            [0, 1, 2, 4, 8, 14, 27, 36, 55, 111],
            [0, 17, 25, 37, 62, 105, 180, 277, 423, 831],
            [0, 2, 5, 11, 22, 42, 69, 118, 227, 643],
            [0, 15, 22, 34, 61, 104, 179, 265, 375, 713],
            [0, 5, 11, 14, 21, 31, 67, 130, 204, 576],
            [0, 7, 13, 21, 46, 83, 134, 250, 387, 711],
            [0, 6, 9, 16, 30, 47, 89, 128, 186, 388],
            [0, 1, 4, 13, 34],
            [0, 9, 17, 31, 60, 95, 171, 280, 484, 848],
            [0, 1, 3, 7, 19, 48],
            [0, 1, 2, 5, 9, 19, 40],
            [0, 1, 3, 7, 13, 27, 66, 99, 176, 482],
            [1, 2, 4, 9, 16, 29, 52, 149],
            [0, 1, 2, 4, 6, 9, 14, 27, 41],
            [0, 1, 3, 7, 13, 29, 71],
            [0, 1, 3, 7, 13, 34],
            [0, 3, 8, 19, 31, 55, 100, 201, 300, 670],
            [0, 1, 3, 6, 12, 19, 36, 82],
            [0, 1, 3, 6, 9, 16, 33],
            [0, 1, 3, 7, 36],
            [0, 1, 3, 7, 12, 20, 33, 70],
            [1, 18, 26, 39, 69, 119, 204, 307, 448, 875],
            [8, 87, 132, 210, 415, 787, 1370, 1904, 2621, 4964],
            [6, 36, 54, 96, 164, 270, 490, 895, 1379, 2787],
            [1, 4, 6, 11, 15, 22, 39, 70, 110, 241],
            [0, 1, 6, 13, 23, 43, 107, 184, 296, 637],
            [0, 22, 36, 64, 119, 197, 331, 553, 887, 1814],
        ]
        self.autophase_dim = 553

        self.benchmarks = Benchmarks(
            args.load_benchmarks_from_json,
            args.json_key,
            args.min_ir,
            args.max_ir,
            split_rank=benchmark_split_rank,
            num_splits=num_splits,
            use_only_anghabench=args.use_only_anghabench,
            num_benchmarks=args.num_benchmarks,
        ) if (traj_db is None and pydantic_file is None) else None
        self.test_result_queue = test_result_queue
        self._make_test_env = make_test_env
        if self.test_result_queue is not None:  # indicates we are using online testing
            assert (
                self.run_locally
            )  # need to collect the model's rowid in the model database
        if args.online_test_json:
            self.test_benchmarks = {
                tag: Benchmarks(
                    args.online_test_json,
                    tag,
                    split_rank=test_split_rank,
                    num_splits=num_splits,
                )
                for tag in ["validation-small", "test-small"]
            }
        self.benchmark_split_rank = benchmark_split_rank
        self.num_splits = num_splits
        self.traj_db = traj_db
        self.pydantic_file = pydantic_file

        self.eps = args.eps  # probability of selecting a random action
        self.max_step = args.max_step_per_job
        self.max_step_per_job = args.max_step_per_job
        self.online_test_max_step = args.online_test_max_step
        self.max_episodes = max_episodes
        self.test_frequency_in_seconds = args.test_frequency_in_seconds
        self.model_capacity = args.model_capacity
        # average the last n scores (a score is the maximum return within a trajectory)
        self.num_last_scores = args.avg_last_n_scores
        self.last_n_scores: np.ndarray = np.zeros(self.num_last_scores, dtype=np.float32)
        self.score_ptr = 0
        self.traj_scores = []
        self.benchmark_metrics = {}

        self.queue_send = queue0
        self.queue_recv = queue1
        self.rank = local_rank

        self.pydantic_val_dataset = None
        if pydantic_val_dataset_path is not None:
            self.pydantic_val_dataset = TrajectoryDataset.load(pydantic_val_dataset_path).benchmark_dataset
        self.pydantic_test_dataset = None
        if pydantic_test_dataset_path is not None:
            self.pydantic_test_dataset = BenchmarkDataset.load(pydantic_test_dataset_path)
        self.simple_generation = simple_generation
        self.rl_online = self.pydantic_file is not None and self.pydantic_val_dataset is not None

        self.alive = True  # Set this to False to signal the thread to stop.
        self.last_test_time = time()

    def run(self) -> None:
        print(f"Start to run rank {self.benchmark_split_rank}")
        env = self._make_env()
        print(f"Finish env init at rank {self.benchmark_split_rank}")
        if (self.run_locally and self.traj_db is None and self.pydantic_file is None) or self.rl_online:
            self.model = self.create_model()
        if self.traj_db is not None:
            assert self.run_locally, "Need to set run_locally=True to avoid blocking code."
            self.traj_con = sqlite3.connect(self.traj_db, timeout=600)
            self.traj_cur = self.traj_con.cursor()
            benchmarks = self.traj_cur.execute("select benchmark_uri from Metrics").fetchall()
            benchmarks = benchmarks[self.benchmark_split_rank::self.num_splits]
            benchmarks = [bm[0] for bm in benchmarks]
            print(f"Using {len(benchmarks)} benchmarks in rank {self.benchmark_split_rank}")
            i = 0
        if self.pydantic_file is not None:
            assert self.run_locally, "Need to set run_locally=True to avoid blocking code."
            if os.path.getsize(self.pydantic_file) / (1024 ** 2) >= 200:
                pkl_file = self.pydantic_file.replace(".json", "_seqidx.pkl")
                print(f"Loading TrajectoryDataset from {pkl_file} at rank {self.benchmark_split_rank}")
                with open(pkl_file, "rb") as f:
                    benchmark2seqidx = pickle.load(f)
                benchmarks = list(benchmark2seqidx.keys())
            else:
                print(f"Loading TrajectoryDataset from {self.pydantic_file} at rank {self.benchmark_split_rank}")
                dataset = TrajectoryDataset.load(self.pydantic_file)

                # for generating states for the best action sequences from the coreset
                self.benchmark2seqir = {
                    item.benchmark : item.all_ir_searches for item in dataset.samples
                }
                self.action_sequences = dataset.action_sequences

                benchmarks = dataset.benchmark_dataset.benchmarks
            benchmarks = benchmarks[self.benchmark_split_rank::self.num_splits]
            print(f"Using {len(benchmarks)} benchmarks in rank {self.benchmark_split_rank}")
            i = 0

        if self.rl_online:
            n_benchmarks = len(benchmarks)
            while self.alive:
                idx = np.random.choice(n_benchmarks)
                benchmark = benchmarks[idx]
                try:
                    self.pydantic_run_one_episode(env, benchmark, self.max_step)
                except Exception as e:
                    print(f"Failed on {benchmark}: {e}")

                # testing
                if (
                    self.test_result_queue is not None
                    and time() - self.last_test_time > self.test_frequency_in_seconds
                ):
                    self.last_test_time = time()  # restart the timer immediately
                    self.pydantic_test_model(self.pydantic_val_dataset, "validation")
                    self.pydantic_test_model(self.pydantic_test_dataset, "test")

            env.close()
            print("Exiting generation process")
            return

        while self.alive:
            if self.traj_db is not None or self.pydantic_file is not None:
                if i >= len(benchmarks):
                    self.alive = False
                    break
                benchmark = benchmarks[i]
                i += 1
            else:
                benchmark = self.benchmarks.sample()
            if benchmark is None:
                break
            if benchmark.find("ghost") >= 0:
                continue
                # tmp = self.max_step
                # self.max_step = 3
            try:
                if self.pydantic_file is not None and not self.simple_generation:
                    # for generating states for the best action sequences from the coreset
                    self.core_act_idx = 0
                    action_ir = self.benchmark2seqir[benchmark]
                    min_ir = min(action_ir)
                    self.good_act_idx = [ai for ai, ir in enumerate(action_ir) if ir == min_ir]
                    for _ in range(len(self.good_act_idx)):
                        self.run_one_episode(env, benchmark)
                else:
                    self.run_one_episode(env, benchmark)
                self.total_episode_count += 1
            except Exception as e:
                print(f"Failed on {benchmark}: {e}")

            # if benchmark.find("ghost") >= 0:
            #     self.max_step = tmp

            # testing
            if (
                self.test_result_queue is not None
                and time() - self.last_test_time > self.test_frequency_in_seconds
            ):
                self.test_model("validation-small")
                self.test_model("test-small")
                self.last_test_time = time()
            
            if self.max_episodes is not None and self.total_episode_count >= self.max_episodes:
                self.alive = False

        env.close()
        if len(self.benchmark_metrics) > 0:
            avg_score = sum(v[0] for k, v in self.benchmark_metrics.items()) / len(
                self.benchmark_metrics
            )
            print(f"avg_score: {avg_score}")

    def test_model(self, tag):
        print(f"[Testing {tag}] started")
        tmp_metrics = self.benchmark_metrics
        self.benchmark_metrics = {}
        test_env = self._make_test_env()
        self.max_step = self.online_test_max_step
        model_rowid = self.model.last_idx
        for benchmark in self.test_benchmarks[tag]:
            try:
                self.run_one_episode(test_env, benchmark)
            except Exception as e:
                print(f"[Testing] Failed on {benchmark}: {e}")
        print(f"[Testing] ended with [{len(self.benchmark_metrics)}] benchmarks")
        if len(self.benchmark_metrics) > 0:
            avg_score = sum(v[0] for k, v in self.benchmark_metrics.items()) / len(
                self.benchmark_metrics
            )
            print(f"Avg score: {avg_score}")
        self.test_result_queue.put_nowait((tag, model_rowid, self.benchmark_metrics))
        # reset things
        self.benchmark_metrics = tmp_metrics
        self.max_step = self.max_step_per_job
        test_env.close()

    def pydantic_test_model(self, dataset: BenchmarkDataset, tag: str):
        print(f"Pydantic testing started")
        t0 = time()
        benchmarks = dataset.benchmarks[self.benchmark_split_rank::self.num_splits]
        test_env = self._make_test_env()
        max_step = self.online_test_max_step + 3
        model_rowid = self.model.last_idx

        all_samples = []
        for benchmark in benchmarks:
            try:
                traj = self.pydantic_run_one_episode(test_env, benchmark, max_step)
                all_samples.append(traj)
            except Exception as e:
                print(f"[Testing] Failed on {benchmark}: {e}")
        print(f"[Testing] ended with [{len(self.benchmark_metrics)}] benchmarks")

        output = PolicyRolloutDataset(
            benchmark_dataset=dataset,
            policy_name='policy_online',
            samples=all_samples)

        self.test_result_queue.put_nowait((tag, model_rowid, output, self.benchmark_split_rank))
        elapse = time() - t0
        print(f"[Testing] used {elapse:.1f} s at rank {self.benchmark_split_rank}")

        test_env.close()

    def _step(self, env, action):
        obs_space = [
            env.observation.spaces[feature_name] for feature_name in self.features
        ]
        reward_spaces = [
            env.reward.spaces[feature_name] for feature_name in self.reward_spaces
        ]
        observation, ir_state, reward, done, _ = env.step(
            action, observation_spaces=obs_space, reward_spaces=reward_spaces
        )
        return dict(zip(self.features, observation)), ir_state, reward, done

    def _get_model_output(
        self, state_id, graph, autophase, current_action_rewards=None
    ):
        if state_id in self.id2output and not self.use_history:
            return self.id2output[state_id]

        if self.use_autophase:
            graph = None
            autophase = self._autophase2vec(autophase)  # .unsqueeze(0)
        else:
            # graph = self.feature_extractor.process_nx_graph(programl)
            # graph = dgl2pyg(graph)
            autophase = None
            if graph.num_nodes > self.model_capacity:
                raise RuntimeError(
                    f"Graph exceeds model capacity: #nodes = {graph.num_nodes} > {self.model_capacity}"
                )
        if self.use_history:
            action_history = torch.tensor(
                [s[0] for s in current_action_rewards], dtype=torch.long
            )
            reward_history = torch.tensor(
                [s[1] for s in current_action_rewards], dtype=torch.float
            )
        else:
            action_history, reward_history = None, None
        job = (self.rank, graph, autophase, action_history, reward_history)
        # Then send to neural network
        if self.run_locally:
            logit, value, repr_ = self.model.get_model_output_local(job)
        else:
            self.queue_send.put(job)
            logit, value, repr_ = self.queue_recv.get()
        if logit is None:
            raise RuntimeError(
                "Failed to compute the model output: got None from queue."
            )

        m = Categorical(logits=logit.flatten())
        self.id2output[state_id] = m, logit, value, repr_
        return m, logit, value, repr_

    def _autophase2vec_0(self, autophase):
        f = torch.zeros(self.autophase_dim)
        n = 0
        for i, (b, x) in enumerate(zip(self.autophase_bounds, autophase)):
            idx = len(b)
            x = int(x)
            for j, th in enumerate(b):
                if x <= th:
                    idx = j
                    break
            f[n + idx] = 1.0
            n += len(b) + 1
        assert (
            n == self.autophase_dim
        ), f"Autophase dimension mismatch! n = {n}, autophase_dim = {self.autophase_dim}"
        return f

    def _autophase2vec(self, raw_state):
        if isinstance(raw_state, str):
            raw_state = [int(x) for x in raw_state.split(" ")]
        autophase = torch.tensor(raw_state, dtype=torch.float)
        # if self.normalize_autophase:
        # normalize autophase instruction counts
        total_instructions = raw_state[51]
        autophase = autophase / total_instructions
        return autophase

    def get_traj(self, benchmark_uri: str):
        assert self.traj_db is not None
        res = self.traj_cur.execute("select best_action_seq from Metrics where benchmark_uri = ?", (benchmark_uri,)).fetchone()
        actions = res[0]
        if len(actions) == 0:
            actions = []
            raise RuntimeError("Empty trajectory.")
        else:
            actions = [int(a) for a in actions.split(" ")]
        return actions

    def _get_action(self, env, observation, ir_state):
        # programl = None if self.use_autophase else observation["Programl"]
        # autophase = observation["Autophase"]
        state_id = observation["IrSha1"]

        if not self.use_random_action and random.random() >= self.eps:
            m, logit, value, repr_ = self._get_model_output(
                state_id, ir_state, ir_state, env.current_action_rewards
            )
            action = m.sample()
            logp = m.log_prob(action).item()
            action = action.item()
        else:
            n_action = env.action_space.n
            action = torch.randint(
                0, n_action, (1,)
            )  # not to use the sampling by compiler_gym env
            value, logp, repr_ = None, None, None
            if self.use_ppo:
                m, logit, value, repr_ = self._get_model_output(
                    state_id, ir_state, ir_state, env.current_action_rewards
                )
                logp = m.log_prob(action).item()
            action = action.item()

        return action, value, logp, repr_

    def restart_episode(self, init_return):
        self.traj_scores = [init_return]
        self.id2output = {}

    def end_episode(self, env, benchmark: str):
        env.on_traj_done()
        score = max(self.traj_scores) / float(self.IrInstructionCountOz)
        self.benchmark_metrics[benchmark] = (
            score,
            max(self.traj_scores),
            self.IrInstructionCountOz,
        )
        self.last_n_scores[self.score_ptr] = score
        self.score_ptr = (self.score_ptr + 1) % self.num_last_scores
        if (
            self.rank == 0
            and self.total_episode_count >= self.num_last_scores
            and self.total_episode_count % (self.num_last_scores // 10) == 0
        ):
            print(
                f"Mean of the improvement over Oz of last {self.num_last_scores} episode: {self.last_n_scores.mean()}",
                flush=True,
            )
            cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            log_text = f"{cur_time}: {self.last_n_scores.mean()}\n"
            write(log_text, "./mean_scores.txt")
        self.total_episode_count += 1

    def get_core_actions(self):
        idx = self.good_act_idx[self.core_act_idx]
        actions = self.action_sequences.actionseqs[idx].actions
        self.core_act_idx += 1
        return actions

    def run_one_episode(self, env: CompilerOnPolicyWrapper, benchmark: str):
        env.reset(benchmark=benchmark)

        self.IrInstructionCountOz = int(env.observation["IrInstructionCountOz"])
        self.IrInstructionCountO0 = int(env.observation["IrInstructionCountO0"])
        self.IrInstructionCountOzReduction = (
            self.IrInstructionCountO0 - self.IrInstructionCountOz
        )
        init_observation, ir_state, init_reward, done = self._step(env, [])
        init_return = self.IrInstructionCountOz - env.observation["IrInstructionCount"]
        next_observation = init_observation

        actions = None
        self.restart_episode(init_return)
        if self.traj_db is not None:
            actions = self.get_traj(benchmark)
        if self.pydantic_file is not None:
            if self.simple_generation:
                actions = [120]  # use a dummy action to make one transition to save into db
            else:
                actions = self.get_core_actions()

        # simply follow the policy by sampling an action
        for i in range(self.max_step):

            if actions is not None:
                if i >= len(actions):
                    break
                action = actions[i]
                value, logp, repr_ = (None,) * 3
            else:
                action, value, logp, repr_ = self._get_action(
                    env, next_observation, ir_state
                )
            next_observation, ir_state, reward, done = self._step(env, action)
            self.traj_scores.append(self.traj_scores[-1] + reward)
            env.record_value(value, logp, repr_)

            if not self.alive or done:
                break
        if done:
            raise RuntimeError("Trajectory ended with done.")

        action, value, logp, repr_ = self._get_action(env, next_observation, ir_state)
        env.record_value(value, logp, repr_)

        self.end_episode(env, benchmark)

    def pydantic_run_one_episode(self, env: CompilerOnPolicyWrapper, benchmark: str, max_step: int):
        env.reset(benchmark=benchmark)

        self.IrInstructionCountOz = int(env.observation["IrInstructionCountOz"])
        self.IrInstructionCountO0 = int(env.observation["IrInstructionCountO0"])
        self.IrInstructionCountOzReduction = (
            self.IrInstructionCountO0 - self.IrInstructionCountOz
        )

        init_observation, ir_state, init_reward, done = self._step(env, [])
        init_return = self.IrInstructionCountOz - env.observation["IrInstructionCount"]
        next_observation = init_observation

        self.restart_episode(init_return)
        all_counts = [self.IrInstructionCountO0]
        for i in range(max_step):

            action, value, logp, repr_ = self._get_action(
                env, next_observation, ir_state
            )
            next_observation, ir_state, reward, done = self._step(env, action)
            self.traj_scores.append(self.traj_scores[-1] + reward)
            env.record_value(value, logp, repr_)
            all_counts.append(int(env.observation["IrInstructionCount"]))

            if not self.alive or done:
                break
        if done:
            raise RuntimeError("Trajectory ended with done.")

        action, value, logp, repr_ = self._get_action(env, next_observation, ir_state)
        env.record_value(value, logp, repr_)

        self.end_episode(env, benchmark)

        traj = PolicyRolloutSample(
            benchmark=benchmark, ir_original=self.IrInstructionCountO0, 
            ir_compiler=self.IrInstructionCountOz, ir_search_trajectory=all_counts)
        return traj


def run_agent(
    job_id: int,
    seed: int,
    args,
    queue0,
    queue1,
    local_rank_in_gpu,
    model_rank,
    nproc,
    test_result_queue,
):
    if args.outdir is not None:
        if args.outdir.endswith(".db"):
            db_path = Path(args.outdir)
        else:
            db_path = Path(os.path.join(args.outdir, f"summary.db"))
        db_path.parent.mkdir(exist_ok=True, parents=True)
        socket_db = Path(os.path.join(args.outdir, f"socket.db"))
    else:
        db_path = None
    vocab_db_path = Path(args.vocab_db_path) if args.vocab_db_path else None
    Wrapper = CompilerPPOWrapper if args.use_ppo else CompilerOnPolicyWrapper

    create_socket = None
    if args.send_data_via_socket:

        def create_socket():
            return DataServer(socket_db=socket_db)

    def make_env():
        env = gym.make("llvm-autophase-ic-v0")
        env = Wrapper(
            env,
            args,
            db_path=db_path,
            vocab_db_path=vocab_db_path,
            online_update_vocab=args.online_update_vocab,
            graph_version=args.graph_version,
            traj_last_n=args.traj_last_n,
            reward_discount=args.reward_discount,
            use_autophase=args.use_autophase,
            return_lower_bound=args.return_lower_bound,
            GAE_lambda=args.GAE_lambda,
            send_data_via_socket=args.send_data_via_socket,
            create_socket=create_socket,
            for_reinforce=args.for_reinforce,
            highest_reward=args.highest_reward,
        )
        return env

    args.online_test_max_step

    if test_result_queue is not None:

        def make_test_env():
            env = gym.make("llvm-autophase-ic-v0")
            env = Wrapper(
                env,
                args,
                db_path=None,
                vocab_db_path=vocab_db_path,
                online_update_vocab=args.online_update_vocab,
                graph_version=args.graph_version,
                traj_last_n=args.traj_last_n,
                reward_discount=args.reward_discount,
                use_autophase=args.use_autophase,
                return_lower_bound=args.return_lower_bound,
                GAE_lambda=args.GAE_lambda,
                for_reinforce=args.for_reinforce,
            )
            return env

    else:
        make_test_env = None

    if args.run_model_locally:

        def create_model():
            return GPUworker(
                queue0,
                [queue1],
                model_rank,
                device=args.device,
                model_db_path=args.model_db_path,
                model_capacity=args.model_capacity,
                load_full_rate=args.load_full_rate,
                job_full_rate=args.job_full_rate,
                wait_time=args.wait_time,
                load_model_frequency=args.load_model_frequency,
                run_locally=args.run_model_locally,
                model_rowid=args.model_rowid,
                eval_model=args.eval_on_policy,
            )

    else:
        create_model = None
    benchmark_split_rank = None
    if args.eval_on_policy or args.traj_db is not None or args.pydantic_datasource is not None:
        benchmark_split_rank = job_id
    worker = AgentWorker(
        job_id,
        seed,
        args,
        queue0,
        queue1,
        local_rank_in_gpu,
        make_env=make_env,
        use_autophase=args.use_autophase,
        use_history=args.use_history,
        run_locally=args.run_model_locally,
        create_model=create_model,
        use_ppo=args.use_ppo,
        benchmark_split_rank=benchmark_split_rank,
        num_splits=nproc,
        test_split_rank=job_id,
        test_result_queue=test_result_queue,
        make_test_env=make_test_env,
        max_episodes=args.max_episodes,
        traj_db=args.traj_db,
        pydantic_file=args.pydantic_datasource,
        pydantic_val_dataset_path=args.pydantic_val_dataset_path,
        pydantic_test_dataset_path=args.pydantic_test_dataset_path,
        simple_generation=args.simple_generation,
    )
    worker.start()

    if args.runtime_per_job is not None:
        sleep(args.runtime_per_job)
        worker.alive = False
        timeout = 300
    else:
        timeout = None

    try:
        worker.join(timeout=timeout)
    except:
        pass
    return worker.benchmark_metrics


def run_agent_packed_args4(job):
    return run_agent(*job)


def run_model(job_id: int, args, queue0, queue1, local_rank):

    worker = GPUworker(
        queue0,
        queue1,
        local_rank,
        device=args.device,
        model_db_path=args.model_db_path,
        model_capacity=args.model_capacity,
        load_full_rate=args.load_full_rate,
        job_full_rate=args.job_full_rate,
        wait_time=args.wait_time,
        load_model_frequency=args.load_model_frequency,
        model_rowid=args.model_rowid,
        eval_model=args.eval_on_policy,
    )
    worker.start()

    if args.runtime_per_job is not None:
        sleep(args.runtime_per_job)
        worker.alive = False
        timeout = 300
    else:
        timeout = None

    try:
        worker.join(timeout=timeout)
    except:
        pass


def run_model_packed_args(job):
    return run_model(*job)

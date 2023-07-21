
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import random
from pathlib import Path
from threading import Thread
from time import sleep, time
from typing import Callable, Iterable, List

import gym
import numpy as np
import torch
from compiler_gym import CompilerEnv
from compiler_gym.util.gym_type_hints import ActionType
from depq import DEPQ
from torch_geometric.data import Batch

from rlcompopt.env_wrapper.parsing_utils import FeatureExtractor
from rlcompopt.env_wrapper.pyg_utils import dgl2pyg
from rlcompopt.env_wrapper.wrapper_offline import CompilerDatasetLoggingWrapper

log = logging.getLogger(__file__)


class Item(object):
    def __init__(
        self,
        curr_return,
        future_pred,
        env,
        observation,
        action_history,
        next_action=None,
    ):
        # curr_return is always the Oz ir count - current ir count
        if not isinstance(curr_return, np.ndarray):
            curr_return = np.array([curr_return], dtype=np.float64)
        self.curr_return = curr_return
        self.future_pred = future_pred
        self.env = env
        self.observation = observation
        self.action_history = action_history
        self.next_action = next_action

    def priority(self):
        return self.curr_return.item() + self.future_pred

    def get_next(self, next_env, action, future_pred, next_observation, reward):
        next_return = reward + self.curr_return
        next_action_history = self.action_history.copy() + [action]
        return Item(
            next_return, future_pred, next_env, next_observation, next_action_history
        )


class ItemAQ(Item):
    """An item with priority for AQ*"""

    def get_next(
        self, next_env, action, future_pred, next_observation, reward, next_action
    ):
        next_return = reward + self.curr_return
        next_action_history = self.action_history.copy() + [action]
        return ItemAQ(
            next_return,
            future_pred,
            next_env,
            next_observation,
            next_action_history,
            next_action,
        )


class AgentWorker(Thread):
    """Worker thread to run a repeating agent.
    To stop the agent, set the :code:`alive` attribute of this thread to False.
    """

    def __init__(
        self,
        job_id: int,
        seed: int,
        benchmark: str,
        make_env: Callable[[], CompilerEnv],
        args,
    ):
        super().__init__()
        self._make_env = make_env
        self._patience = args.patience
        self.reset_best_return_on_every_episode = (
            args.reset_best_return_on_every_episode
        )
        # self.get_ir_Oz_reduction = args.get_ir_Oz_reduction

        # Incremental progress.
        self.total_environment_count = 0
        self.total_episode_count = 0
        self.total_step_count = 0
        self.best_returns = np.array([0.0], dtype=np.float64)
        self.reward_spaces = ["IrInstructionCount"]
        self.best_actions: List[ActionType] = []
        self.best_commandline: str = []
        self.best_found_at_time = time()
        self.IrInstructionCountOzReduction = None
        self.num_states = 0
        self.num_covered_states = 0

        self.features = [
            "IrSha1",
            "Programl",
            "Autophase",
            "InstCount",
            "IrInstructionCount",
        ]
        self.feature_extractor = args.vocab_db_path and FeatureExtractor(
            args.vocab_db_path, graph_version=args.eval_model.graph_version
        )

        self.benchmark = benchmark
        self.divided_by_this_ir = args.divided_by_this_ir

        if args.gpu is not None and isinstance(args.gpu, str):
            args.gpu = args.gpu.split(",")
        if args.gpu is not None and not isinstance(args.gpu, Iterable):
            args.gpu = [args.gpu]
        self.device = torch.device(
            f"cuda:{args.gpu[job_id % len(args.gpu)]}"
            if args.gpu is not None
            else "cpu"
        )

        # load model
        self.model = args.model_path and torch.load(
            args.model_path, map_location=self.device
        )
        if self.model is not None and not hasattr(self.model, "node_level_action"):
            self.model.node_level_action = False  # this should be deprecated. We use graph-/funciton-/block-level Q learning
        if self.model is not None and not hasattr(self.model, "use_subgraph_feature"):
            self.model.use_subgraph_feature = False
        if self.model is not None and not hasattr(self.model, "divided_by_this_ir"):
            self.model.divided_by_this_ir = False
        if self.model is not None:
            if not hasattr(self.model, "use_fc"):
                self.model.use_fc = False

        if self.model is not None:
            self.model.eval()
            self.gamma = self.model.gamma if hasattr(self.model, "gamma") else 1.0
            if self.model.use_autophase:
                self.autophase_bounds = (
                    self.model.autophase_bounds
                    if hasattr(self.model, "autophase_bounds")
                    else None
                )
                self.autophase_dim = (
                    self.model.autophase_dim
                    if hasattr(self.model, "autophase_dim")
                    else None
                )
            if not hasattr(self.model, "use_relu"):
                self.model.use_relu = False
        self.eps = args.eps
        self.T = args.T
        self.best_n = args.best_n
        self.use_Astar = args.use_Astar
        self.use_AQ = args.use_AQ
        self.use_policy = args.use_policy
        log.info(
            f"Loading vocab = {args.vocab_db_path}, model = {args.model_path}, "
            f"device = {self.device}, eps = {self.eps}, T = {self.T}"
        )

        self.max_step = args.max_step_per_job
        # record the return history.
        self.return_history = []

        # Send random seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        self.seed = seed

        self.alive = True  # Set this to False to signal the thread to stop.

    @property
    def should_run_one_episode(self) -> bool:
        """Whether to run an episode."""
        return self.alive or not self.total_episode_count

    def run(self) -> None:
        """Run episodes in an infinite loop."""
        while self.should_run_one_episode:
            self.total_environment_count += 1
            with self._make_env() as env:
                self._patience = self._patience or env.action_space.n
                self.run_one_environment(env)

    def run_one_environment(self, env: CompilerEnv) -> None:
        """Run random walks in an infinite loop. Returns if the environment ends."""
        while self.should_run_one_episode:
            self.run_one_episode(env)

    # The next two functions can be made a decorator
    def _step(self, env, action):
        obs_space = [
            env.observation.spaces[feature_name] for feature_name in self.features
        ]
        reward_spaces = [
            env.reward.spaces[feature_name] for feature_name in self.reward_spaces
        ]
        observation, reward, done, _ = env.step(
            action, observation_spaces=obs_space, reward_spaces=reward_spaces
        )
        return dict(zip(self.features, observation)), reward, done

    def _get_future_value(self, env, observation, best_n=1):
        # It might return multiple actions/values and they will be sent to the priority queues.
        programl = observation["Programl"]
        autophase = observation["Autophase"]
        preds = self._get_model_output(programl, autophase)
        values, actions = preds.sort(descending=True)
        best_value = values[:best_n].mean().item()
        return best_value

    def _get_model_output(self, programl, autophase):
        if self.model.use_autophase:
            graph = None
            autophase = self._autophase2vec(autophase).unsqueeze(0)
            autophase = autophase.to(self.device)
        else:
            graph = self.feature_extractor.process_nx_graph(programl)
            if self.model.mode == "pyg":
                graph = Batch.from_data_list([dgl2pyg(graph)])
            graph = graph.to(self.device)
            autophase = None
        # Then send to neural network
        preds, _ = self.model(graph, autophase=autophase)
        preds = preds.detach().squeeze()
        return preds

    def _autophase2vec(self, autophase):
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

    def _get_action_value(self, env, observation, best_n=1):
        # It might return multiple actions/values and they will be sent to the priority queues.
        programl = observation["Programl"]
        autophase = observation["Autophase"]

        not_use_random_action = (
            self.feature_extractor is not None
            and self.model is not None
            and random.random() >= self.eps
        )

        if not_use_random_action:
            assert not isinstance(programl, np.int64)
            preds = self._get_model_output(programl, autophase)
            # Get the action back
            if best_n is None:
                # return all actions and future predictions
                qs = preds.tolist()
                actions = list(range(preds.shape[0]))
                return actions, qs
            if self.T == 0:
                _, actions = preds.sort(descending=True)
                actions = actions[:best_n]
                actions = actions.tolist()
            else:
                # sample without replacement
                preds = preds / self.T
                preds = preds.softmax(
                    dim=-1
                )  # must use softmax to make the temperature take effect
                actions = torch.multinomial(
                    preds, num_samples=self.best_n, replacement=False
                ).tolist()

            qs = preds[actions].tolist()
        else:
            actions = [env.action_space.sample() for i in range(best_n)]
            qs = [0] * best_n

        return actions, qs

    def restart_episode(self, init_return):
        if self.reset_best_return_on_every_episode:
            self.best_returns = np.array([init_return], dtype=np.float64)
        self.curr_patience = self._patience
        self.curr_return_history = []

    def end_episode(self):
        if len(self.curr_return_history) > 0:
            self.return_history.append(self.curr_return_history)
            self.total_episode_count += 1

    def save_history(self, item, step=True):
        if step:
            self.total_step_count += 1

        self.curr_return_history.append(
            dict(
                curr_return=item.curr_return.item(),
                future_pred=item.future_pred,
                action_history=item.action_history,
            )
        )

        # print(item.curr_return, future_pred, next_action_history)
        # print(priority, item)
        if (item.curr_return > self.best_returns).any():
            self.curr_patience = self._patience
            self.best_returns = item.curr_return
            self.best_actions = item.action_history
            try:
                self.best_commandline = item.env.commandline()
            except NotImplementedError:
                self.best_commandline = ""
            self.best_found_at_time = time()

        if self.total_step_count >= self.max_step:
            # Exiting.
            self.alive = False

    def run_one_episode(self, init_env: CompilerEnv) -> bool:
        """Run a single random episode.
        :param env: An environment.
        :return: True if the episode ended gracefully, else False.
        """
        init_env.reset()

        self.IrInstructionCountOz = int(init_env.observation["IrInstructionCountOz"])
        self.IrInstructionCountO0 = int(init_env.observation["IrInstructionCountO0"])
        self.IrInstructionCountOzReduction = (
            self.IrInstructionCountO0 - self.IrInstructionCountOz
        )
        init_observation, init_reward, done = self._step(init_env, [])
        init_return = self.IrInstructionCountOz - init_observation["IrInstructionCount"]
        item = Item(init_return, 0, init_env, init_observation, [])

        self.restart_episode(init_return)
        if not self.use_AQ:
            self.save_history(item, step=False)

        if self.use_Astar:
            # we still need a priority queue even if best_n==1 and using A*
            pq = DEPQ(iterable=None, maxlen=10000)

            pq.insert(item, item.priority())

            while self.curr_patience >= 0 and pq.size() > 0:
                self.curr_patience -= 1
                item, priority = pq.popfirst()

                # Next steps.
                actions, future_preds = self._get_action_value(
                    item.env, item.observation, best_n=self.best_n
                )
                for action, future_pred in zip(actions, future_preds):
                    # print("Before forking")
                    next_env = item.env.fork()
                    # print("After forking")
                    next_observation, reward, done = self._step(next_env, action)
                    if self.model is not None:
                        # if we have a model, follow the way in which the model predicts future return
                        pred0 = self._get_future_value(
                            item.env, next_observation, best_n=1
                        )  # average over best `n` future predictions
                        if (
                            self.model is not None and self.model.divided_by_this_ir
                        ) or self.divided_by_this_ir:  # this ir is the ir count of the input state to GNN
                            divider = next_observation["IrInstructionCount"]
                        else:
                            divider = self.IrInstructionCountO0
                        future_pred = (
                            pred0 * divider
                        )  # * self.gamma  # gamma controls how much we trust the future prediction
                    item_ = item.get_next(
                        next_env, action, future_pred, next_observation, reward
                    )

                    self.save_history(item_)
                    if not self.alive:
                        break

                    # print("After Stepping")
                    if not done:
                        # low priority object will be replaced automatically.
                        pq.insert(item_, item_.priority())

                if not self.alive:
                    break

        elif self.use_AQ:
            # use the AQ* as in https://arxiv.org/pdf/2102.04518.pdf
            pq = DEPQ(iterable=None, maxlen=100000)
            item = ItemAQ(
                init_return,
                0,
                init_env,
                init_observation,
                action_history=[],
                next_action=[],
            )
            pq.insert(item, item.priority())

            self.curr_patience += 1  # the first action is [], so not counting it

            closed = (
                {}
            )  # save the state id of the states that were pushed into the queue

            while self.curr_patience >= 0 and pq.size() > 0:
                self.curr_patience -= 1
                item, priority = pq.popfirst()

                curr_env = item.env.fork()
                curr_action = item.next_action
                next_observation, reward, done = self._step(curr_env, curr_action)

                curr_return = item.curr_return + reward
                state_id = " ".join(
                    str(x_) for x_ in next_observation["Autophase"]
                )  # next_observation["IrSha1"]

                if state_id not in closed.keys() or curr_return > closed[state_id]:
                    closed[state_id] = curr_return

                    # expand the node using a model
                    actions, future_preds = self._get_action_value(
                        curr_env, next_observation, best_n=None
                    )

                    if (
                        self.model is not None and self.model.divided_by_this_ir
                    ) or self.divided_by_this_ir:  # this ir is the ir count of the input state to GNN
                        divider = next_observation["IrInstructionCount"]
                    else:
                        divider = self.IrInstructionCountO0

                    for next_action, pred0 in zip(actions, future_preds):
                        pred = (
                            pred0 * divider
                        )  # * self.gamma  # gamma controls how much we trust the future prediction
                        item_ = item.get_next(
                            curr_env,
                            curr_action,
                            pred,
                            next_observation,
                            reward,
                            next_action,
                        )
                        pq.insert(item_, item_.priority())

                item_ = item.get_next(
                    curr_env, curr_action, 0, next_observation, reward, -1
                )
                self.save_history(item_)  # saving one of the `item_` will suffice
                if not self.alive:
                    break

        elif self.use_policy:
            # simply follow the policy with either argmax or sampling (with a temperature)
            while self.curr_patience >= 0:
                self.curr_patience -= 1

                actions, future_preds = self._get_action_value(
                    item.env, item.observation, best_n=1
                )
                action = actions[0]
                next_observation, reward, done = self._step(item.env, action)

                item = item.get_next(item.env, action, 0, next_observation, reward)
                self.save_history(item)
                if not self.alive or done:
                    break

        else:
            while self.curr_patience >= 0:
                self.curr_patience -= 1
                # always pick the best one in this case.
                actions, future_preds = self._get_action_value(
                    item.env, item.observation, best_n=self.best_n
                )
                action = actions[0]
                future_pred = future_preds[0]

                next_observation, reward, done = self._step(item.env, action)
                item = item.get_next(
                    item.env, action, future_pred, next_observation, reward
                )

                self.save_history(item)
                if not self.alive:
                    break

        self.end_episode()


def get_benchmarks(env, dataset_names=None, exclude=None) -> List[str]:
    """Enumerate benchmark URIs to use."""
    benchmarks = []
    if isinstance(dataset_names, str):
        dataset_names = dataset_names.split(",")
    for b in dataset_names:
        benchmarks += list(env.datasets[b].benchmark_uris())
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = exclude.split(",")
        for item in exclude:
            benchmarks = [b for b in benchmarks if item not in b]
    return benchmarks


def load_benchmarks_from_json(json_file, key, return_raw=False):
    with open(json_file, "r") as f:
        data = json.load(f)
    benchmarks = []
    for benchmark, v in data.items():
        benchmarks.extend(v[key])
    if return_raw:
        return benchmarks
    if not isinstance(benchmarks[0], str):
        if isinstance(benchmarks[0], (list, tuple)):
            benchmarks = [b[1] for b in benchmarks]
        elif isinstance(benchmarks[0], dict):
            benchmarks = [b["benchmark"] for b in benchmarks]
        else:
            raise ValueError(
                f"Cannot recognize format of type {type(benchmarks[0])}: {benchmarks[0]}"
            )
    return benchmarks


def load_datasets_from_json(json_file, key, min_ir, max_ir, num_benchmarks=None):
    with open(json_file, "r") as f:
        data = json.load(f)
    benchmarks = {}
    for benchmark, v in data.items():
        bm = v[key]  # a list of [ir_count, benchmark_name]
        filtered_bm = [b for i, b in bm if min_ir <= i <= max_ir]
        if filtered_bm:
            assert isinstance(bm[0][0], int) and isinstance(bm[0][1], str)
            if num_benchmarks is not None:
                rng = np.random.default_rng(0)
                idx = rng.choice(len(filtered_bm), num_benchmarks, replace=False).tolist()
                filtered_bm = [filtered_bm[i] for i in idx]
            # only use non-empty ones
            benchmarks[benchmark] = filtered_bm
    return benchmarks


def run_agent(job_id: int, benchmark: str, seed: int, args):
    """Run random search for a fixed amount of time on a single benchmark."""
    # benchmark format: e.g., "benchmark://cbench-v1/qsort"
    def make_env():
        env = gym.make("llvm-autophase-ic-v0", benchmark=benchmark)
        if args.outdir is not None:
            name = "-".join(benchmark.replace("benchmark://", "").split("/"))
            db_path = os.path.join(args.outdir, f"{name}-{job_id:03d}.db")
            vocab_db_path = Path(args.vocab_db_path) if args.vocab_db_path else None
            env = CompilerDatasetLoggingWrapper(
                env,
                Path(db_path),
                vocab_db_path,
                online_update_vocab=args.online_update_vocab,
                graph_version=args.graph_version,
            )
        return env

    worker = AgentWorker(
        job_id, seed, benchmark, make_env=lambda i=job_id: make_env(), args=args
    )
    worker.start()
    # worker.run()

    if args.runtime_per_job is not None:
        sleep(args.runtime_per_job)
        worker.alive = False
        timeout = 300
    else:
        # Then we use max_step to control
        timeout = None

    try:
        worker.join(timeout=timeout)
    except:  # noqa
        # Service error can be raised on abrupt service termination causing
        # RPC errors.
        pass

    covered_rate = (
        worker.num_covered_states / float(worker.num_states)
        if worker.num_states != 0
        else 0
    )
    return (
        benchmark,
        seed,
        worker.total_step_count,
        float(worker.best_returns.item()),
        worker.IrInstructionCountOz,
        worker.best_actions,
        worker.best_commandline,
        worker.return_history,
        covered_rate,
    )


def run_agent_packed_args(job):
    return run_agent(*job)

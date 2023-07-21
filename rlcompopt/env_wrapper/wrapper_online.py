
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import pickle
import random
import sqlite3
import string
import zlib
from pathlib import Path
from time import time
from typing import Callable

import humanize
import numpy as np
import zmq
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers import CompilerEnvWrapper
from scipy import signal

from rlcompopt.cl.data_socket import DataServer
from rlcompopt.env_wrapper.pyg_utils import dgl2pyg

from .parsing_utils import FeatureExtractor

log = logging.getLogger(__file__)

with open(os.path.join(os.path.dirname(__file__), "database_schema4.sql")) as f:
    DB_CREATION_SCRIPT = f.read()


def unique_list(a_list):
    seen = set()
    result = []
    for x in a_list:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result


def to_str(values):
    return " ".join(str(x) for x in values)


class CompilerOnPolicyWrapper(CompilerEnvWrapper):
    """A wrapper for an LLVM environment that logs all transitions to an sqlite
    database.
    """

    def __init__(
        self,
        env: LlvmEnv,
        args: argparse.Namespace,
        db_path: Path,
        vocab_db_path: Path = None,
        online_update_vocab=False,
        graph_version=0,
        traj_last_n=5,
        reward_discount=0.9,
        use_autophase=False,
        return_lower_bound=-1,
        GAE_lambda=0.97,
        send_data_via_socket=False,
        create_socket: Callable[[], DataServer] = None,
        for_reinforce=True,
        highest_reward=False,  # use the highest reward from the current step to the end
    ):
        super().__init__(env)
        assert isinstance(self.unwrapped, LlvmEnv), "Requires LlvmEnv base environment"

        # db_path.parent.mkdir(exist_ok=True, parents=True)
        self.db_path = db_path

        self.vocab_db_path = vocab_db_path
        self.online_update_vocab = online_update_vocab
        self.graph_version = graph_version

        self.max_buffer_length = args.max_state_buffer_length
        self.commit_frequency = args.commit_frequency_in_seconds
        self.norm_reward = args.norm_reward

        self.traj_last_n = traj_last_n
        self.reward_discount = reward_discount
        self.use_autophase = use_autophase
        self.return_lower_bound = return_lower_bound
        self.send_data_via_socket = send_data_via_socket
        self.lam = GAE_lambda
        self.for_reinforce = for_reinforce

        # these are the buffer to save to the database
        self.transition_buffer = []
        self.state_buffer = []
        self.traj_buffer = []

        # these are the buffer of the current trajectory
        self.current_raw_transitions = []  # for saving the complete trajectory
        self.current_transitions = []  # for saving the transistion for training
        self.time_stamps = []
        self.current_states = []
        self.current_action_rewards = []
        self.state_dict = {}  # for aggregating transitions for individual states
        self.id2state = {}  # query graph given state_id
        self.step_count = 0
        self.highest_reward = highest_reward

        self.features = ["IrSha1"]
        # self.features = ["IrSha1", "Programl", "Autophase", "IrInstructionCount"]
        self.num_actions = self.action_space.n

        self._init_sql()
        self.last_commit = time()

        if self.send_data_via_socket:
            self.data_tunnel = create_socket()

    def _init_sql(self):
        if self.db_path is not None:
            self.connection = sqlite3.connect(self.db_path, timeout=3200)
            self.cursor = self.connection.cursor()
            self.cursor.executescript(DB_CREATION_SCRIPT)
            self.connection.commit()
        self.feature_extractor = FeatureExtractor(
            self.vocab_db_path,
            online_update=self.online_update_vocab,
            graph_version=self.graph_version,
        )

    def reset(self, benchmark: str, *args, **kwargs):
        actions = []
        # could be a specical benchmark uri of the form 'benchmark://poj104-v1/58/1121_ir_95_actions_81_98_59_50_39'
        benchmark0 = benchmark
        if "_ir_" in benchmark and "_actions_" in benchmark:
            benchmark, _, others = benchmark.partition("_ir_")
            ir, _, actions = others.partition("_actions_")
            ir = int(ir)
            actions = [int(a) for a in actions.split("_")] if actions else []

        observation = self.env.reset(benchmark, *args, reward_space="IrInstructionCount", **kwargs)

        if actions:
            observation, reward, done, info = self.env.multistep(actions=actions)
            if done:
                raise RuntimeError("Got done after multistep")

        self.benchmark_uri = benchmark0

        self.current_raw_transitions = []  # for saving the complete trajectory
        self.current_transitions = []
        self.time_stamps = []
        self.current_states = []
        self.current_action_rewards = []
        self.current_values = []
        self.current_logps = []
        self.current_repr = []
        self.state_dict = {}
        self.id2state = {}

        self.step_count = 0
        self.prev_state_id = None
        self._observations = [
            self.env.observation.spaces[feat] for feat in self.features
        ]
        self._rewards = [self.env.reward.spaces["IrInstructionCount"]]
        self._reward_totals = np.zeros(len(self._rewards))
        return observation

    def _split(self, mapping, s1, s2):
        r1 = [mapping[repr(s)] for s in s1]
        r2 = [mapping[repr(s)] for s in s2]
        return r1, r2

    def _split_reward(self, rewards):
        n_internal_reward = len(self._rewards)
        internal_reward = rewards[:n_internal_reward]
        external_reward = rewards[n_internal_reward:]
        return internal_reward, external_reward

    def step(self, action, **kwargs):
        assert self.observation_space, "No observation space set"
        assert self.reward_space, "No reward space set"

        kwargs2 = dict(kwargs)
        if "observation_spaces" in kwargs2:
            del kwargs2["observation_spaces"]

        if "reward_spaces" in kwargs2:
            del kwargs2["reward_spaces"]

        overall_obs_spaces = unique_list(
            self._observations + [self.observation_space_spec]
        )
        overall_reward_spaces = unique_list(self._rewards + [self.reward_space])

        # reset can be a special action, its index is self.num_actions
        if action == self.num_actions:
            assert len(overall_reward_spaces) == 1, f"Only support giving the reward 'IrInstructionCount', but got {overall_reward_spaces}"
            ir_old = self.env.observation["IrInstructionCount"]
            self.env.reset(self.benchmark_uri, reward_space="IrInstructionCount")
            observations, _, done, info = self.env.step(
                action=[],
                observation_spaces=overall_obs_spaces,
                reward_spaces=overall_reward_spaces,
                **kwargs2,
            )
            ir_new = self.env.observation["IrInstructionCount"]
            rewards = (ir_old - ir_new,)

        else:
            observations, rewards, done, info = self.env.step(
                action=action,
                observation_spaces=overall_obs_spaces,
                reward_spaces=overall_reward_spaces,
                **kwargs2,
            )

        observation_mapping = {
            repr(k): v for k, v in zip(overall_obs_spaces, observations)
        }
        reward_mapping = {repr(k): v for k, v in zip(overall_reward_spaces, rewards)}

        external_obs_space = kwargs.get(
            "observation_spaces", [self.observation_space_spec]
        )
        external_reward_space = kwargs.get(
            "reward_spaces", [self.reward_space]
        )  # checked, it is [IrInstructionCountOz]

        # Properly handling observation/reward spaces.
        internal_obs, external_obs = self._split(
            observation_mapping, self._observations, external_obs_space
        )
        internal_rewards, external_rewards = self._split(
            reward_mapping, self._rewards, external_reward_space
        )

        self._reward_totals += internal_rewards

        ir_state = self.record_state(self.actions, internal_obs, done)
        state_id = internal_obs[0]

        assert isinstance(action, int) or action == []
        if isinstance(action, int):
            # only start recording after the first empty action '[]'
            assert self.prev_state_id is not None
            self.record_transition(
                self.prev_state_id,
                action,
                state_id,
                done,
                internal_obs,
                internal_rewards,
                info,
            )
        self.prev_state_id = state_id

        if len(external_obs) == 1:
            external_obs = external_obs[0]

        if len(external_rewards) == 1:
            external_rewards = external_rewards[0]

        return external_obs, ir_state, external_rewards, done, info

    def record_state(self, actions, observations, done) -> None:
        (state_id,) = observations
        if state_id in self.state_dict:
            return self.id2state[state_id]

        if self.use_autophase:
            graph = None
            num_nodes = 1
            autophase = self.observation["Autophase"]
            self.id2state[state_id] = autophase
        else:
            programl = self.observation["Programl"]
            dgl_graph = self.feature_extractor.process_nx_graph(programl)
            graph = dgl2pyg(dgl_graph)
            self.id2state[state_id] = graph
            num_nodes = graph.num_nodes
            autophase = []
        IrInstructionCount = self.observation["IrInstructionCount"]

        state = (
            self.benchmark_uri,
            state_id,
            graph if self.send_data_via_socket else zlib.compress(pickle.dumps(graph)),
            autophase if self.send_data_via_socket else to_str(autophase),
            int(IrInstructionCount),
            num_nodes,
        )

        self.current_states.append(state)
        self.state_dict[state_id] = state
        return graph or autophase

    def record_transition(
        self, state_id, action, next_state_id, done, observations, rewards, info
    ) -> None:
        next_state_id = observations[0]
        IrInstructionCount = float(rewards[0])

        self.current_raw_transitions.append(
            [
                self.benchmark_uri,
                state_id,
                action,
                IrInstructionCount,  # raw reward
                0,  # placeholder for cumulative reward
                next_state_id,
                self.state_dict[state_id][5],
            ]
        )
        # TODO: normalize by O0 or current ir count
        # self.current_action_rewards.append((action, IrInstructionCount / self.state_dict[state_id][4]))
        self.current_action_rewards.append(
            (action, IrInstructionCount / self.current_states[0][4])
        )
        self.time_stamps.append(time())

    def on_traj_done(self):
        traj = self.current_raw_transitions
        n = len(traj)

        # we cut the last part (as their cumulative rewards are not accurate)
        if self.traj_last_n is not None and (n == 0 or n <= self.traj_last_n or self.db_path is None):
            return

        rewards = np.array([traj[i][3] for i in range(n)], dtype=np.float32)
        cum_rewards = discount_cumsum(rewards, self.reward_discount)

        # Divide them by the IrInstructionCount of the previous state and write them to the placeholder.
        # prev_state_ir = np.array([s[4] for s in self.current_states], dtype=np.float)[:-1]
        prev_state_ir = ir_O0 = float(self.current_states[0][4])
        if self.return_lower_bound is not None:
            pass
        if self.for_reinforce:
            divided_cum_rewards = cum_rewards / prev_state_ir
            divided_cum_rewards = divided_cum_rewards - divided_cum_rewards.mean()

        # normalize the rewards to have std 1
        if self.norm_reward:
            assert self.for_reinforce
            reward_std = divided_cum_rewards.std()
            if reward_std.item() == 0:
                raise RuntimeError
                reward_std = 1e-5
            divided_cum_rewards = divided_cum_rewards / reward_std

        if self.for_reinforce:
            cum_rewards = divided_cum_rewards
        else:
            # just record the simple cum_rewards
            pass

        # put them back to the placeholder
        for i in range(n):
            traj[i][4] = cum_rewards[i].item()

        # for k, v in self.state_dict.items():
        #     self.state_dict[k] = [v[0], v[1], [], [], [], [], v[-1]]

        self.current_transitions = traj[: -self.traj_last_n] if self.traj_last_n is not None else traj

        # add traj_id and traj_step
        self.traj_id = self.current_states[0][1] + str_generator(5)
        self.current_transitions = [
            traj + [self.traj_id, i, None, None, self.time_stamps[i]]
            for i, traj in enumerate(self.current_transitions)
        ]

        self.state_buffer.extend(self.current_states)
        self.transition_buffer.extend(self.current_transitions)
        self.save_traj(traj)

        self.flush()

    def save_traj(self, traj):
        state_id_seq = [tr[1] for tr in traj] + [traj[-1][5]]
        action_seq = [tr[2] for tr in traj]
        reward_seq = [rw for _, rw in self.current_action_rewards]
        if self.send_data_via_socket:
            graph_repr = self.current_repr if self.current_repr[0] is not None else None
        else:
            state_id_seq = to_str(state_id_seq)
            action_seq = to_str(action_seq)
            reward_seq = pickle.dumps(reward_seq)
            graph_repr = (
                pickle.dumps(self.current_repr)
                if self.current_repr[0] is not None
                else None
            )

        traj_seq = (
            self.traj_id,
            self.benchmark_uri,
            state_id_seq,  # state id seq (including first observed state and last observed state)
            action_seq,  # action seq
            reward_seq,  # reward seq
            graph_repr,  # all observed state's representation
        )

        self.traj_buffer.append(traj_seq)

    def flush(self, force=False) -> None:
        """Flush the buffered steps and observations to database."""
        if (
            not force
            and len(self.transition_buffer) < self.max_buffer_length
            and time() - self.last_commit < self.commit_frequency
            # and not self.send_data_via_socket
        ) or self.db_path is None:
            # Not yet need flushing
            return

        n_states, n_observations, n_traj = (
            len(self.state_buffer),
            len(self.transition_buffer),
            len(self.traj_buffer),
        )

        # Nothing to flush.
        if not n_observations:
            return

        if self.send_data_via_socket:
            with Timer() as flush_time:
                data = {
                    "States": {
                        state[1]: state for state in self.state_buffer
                    },  # state_id: state
                    "Transitions": {
                        str_generator(20): trans for trans in self.transition_buffer
                    },  # random traj id
                    "Trajectories": {traj[0]: traj for traj in self.traj_buffer},
                }
                self.data_tunnel.send_pyobj(
                    data, flags=zmq.NOBLOCK
                )  # num_data_in_buffer =
            print(
                f"Wrote {n_states} states, {n_observations} transitions, {n_traj} trajectories in {flush_time}. "
                f"Last flush {humanize.naturaldelta(time() - self.last_commit)} ago. "
                # f"The data receiver has {num_data_in_buffer} batches to process."
            )
            self.last_commit = time()
            # while num_data_in_buffer > 2:
            #     sleep(5)
            #     num_data_in_buffer = self.data_tunnel.check_receipt()
            self.state_buffer = []
            self.transition_buffer = []
            self.traj_buffer = []
            return

        with Timer() as flush_time:
            try:
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO States VALUES (?, ?, ?, ?, ?, ?)",
                    self.state_buffer,
                )

                self.cursor.executemany(
                    "INSERT OR IGNORE INTO Transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self.transition_buffer,
                )

                self.cursor.executemany(
                    "INSERT OR IGNORE INTO Trajectories VALUES (?, ?, ?, ?, ?, ?)",
                    self.traj_buffer,
                )

                self.connection.commit()
            except Exception as e:
                # rollback and re-raise the exception
                self.cursor.execute("ROLLBACK")
                self.last_commit = time()
                raise RuntimeError(f"Failed to flush to database: {e}")
            finally:
                self.state_buffer = []
                self.transition_buffer = []
                self.traj_buffer = []

        # synchronize with the trainer
        # result = list(self.cursor.execute("SELECT rowid FROM Transitions ORDER BY rowid DESC LIMIT 1"))
        # if result:
        #     inserted_rows = result[0][0]
        #     while True:
        #         rec = list(self.cursor.execute("SELECT read_rows, num_rows_per_iter FROM TrainerProgress ORDER BY rowid DESC LIMIT 1"))
        #         if rec:
        #             trainer_read_rows, trainer_speed = rec[0]
        #             if inserted_rows - trainer_read_rows - 3 * trainer_speed < 0:
        #                 break
        #             sleep(5)
        #         else:
        #             break

        # NOTE: logging is not working with multiprocessing spawn
        # so there would be no log
        print(
            f"Wrote {n_states} state records, {n_observations} transitions, {n_traj} trajectories in {flush_time}. "
            f"Last flush {humanize.naturaldelta(time() - self.last_commit)} ago"
        )
        self.last_commit = time()

    def close(self):
        self.flush(force=True)
        self.env.close()

    def fork(self):
        raise NotImplementedError

    def record_value(self, value, logp, repr_=None):
        if value is not None:
            value = value.item()
        self.current_values.append(value)
        self.current_logps.append(logp)
        self.current_repr.append(repr_)


class CompilerPPOWrapper(CompilerOnPolicyWrapper):
    def finish_path(self, rew_buf, values, state_ir):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        val_buf = (
            values * state_ir
        )  # get the estimates in the form of ir instruction count

        last_val = val_buf[-1]
        rews = np.append(rew_buf, last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.reward_discount * val_buf[1:] - val_buf[:-1]
        adv_buf = discount_cumsum(deltas, self.reward_discount * self.lam)
        # normalize the advantages to have mean zero and std one
        # adv_buf = (adv_buf - adv_buf.mean()) / adv_buf.std()
        adv_buf /= state_ir[0]

        # the next line computes rewards-to-go, to be targets for the value function
        ret_buf = discount_cumsum(rews, self.reward_discount)[:-1]
        # Divide them by the IrInstructionCount of the previous state
        ret_buf = ret_buf / state_ir[:-1]

        return ret_buf, adv_buf

    def finish_path2(self, rew_buf, values, state_ir):
        """
        For the return signal, use highest reward from the current step to the end.
        """

        val_buf = (
            values * state_ir
        )  # get the estimates in the form of ir instruction count

        cum_rew = np.cumsum(rew_buf)
        cum_rew = np.concatenate([[0], cum_rew])
        return_to_go = np.zeros_like(cum_rew)
        for i in range(1, len(cum_rew)):
            return_to_go[i] = np.max(cum_rew[i:]) - cum_rew[i - 1]

        val_buf[-1] = 0  # the last state has value zero
        adv_buf = rew_buf + self.reward_discount * val_buf[1:]

        # normalize the advantages to have mean zero and std one
        # adv_buf = (adv_buf - adv_buf.mean()) / adv_buf.std()
        adv_buf /= state_ir[0]

        # the next line computes rewards-to-go, to be targets for the value function
        ret_buf = return_to_go[1:]
        # Divide them by the IrInstructionCount of the previous state
        ret_buf = ret_buf / state_ir[:-1]

        return ret_buf, adv_buf

    def on_traj_done(self):
        traj = self.current_raw_transitions
        n = len(traj)

        if n == 0:
            return

        rewards = np.array([traj[i][3] for i in range(n)], dtype=np.float32)
        state_ir = np.array([self.state_dict[s[1]][4] for s in traj] + [self.state_dict[traj[-1][5]][4]], dtype=np.float32)  # include last state ir
        vals = np.array(self.current_values, dtype=np.float32)

        if self.highest_reward:
            ret_buf, adv_buf = self.finish_path2(rewards, vals, state_ir)
        else:
            ret_buf, adv_buf = self.finish_path(rewards, vals, state_ir)

        # put them back to the placeholder
        for i in range(n):
            traj[i][4] = ret_buf[i].item()

        # add traj_id, traj_step, advantage, logp
        self.traj_id = self.current_states[0][1] + str_generator(5)
        traj = [
            tr + [self.traj_id, i, adv_buf[i].item(), self.current_logps[i], self.time_stamps[i]]
            for i, tr in enumerate(traj)
        ]

        self.state_buffer.extend(self.current_states)
        self.transition_buffer.extend(traj)
        self.save_traj(traj)

        self.flush()


CHARS = string.ascii_lowercase + string.digits


def str_generator(size=5, chars=CHARS):
    return "".join(random.SystemRandom().choice(chars) for _ in range(size))


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

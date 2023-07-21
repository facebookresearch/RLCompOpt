
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import copy
import pickle
import sqlite3
import zlib
import string
import random
from pathlib import Path
from time import time

from typing import List, Optional 

import humanize
import numpy as np
from compiler_gym.envs import LlvmEnv
from compiler_gym.util.timer import Timer
from compiler_gym.wrappers import CompilerEnvWrapper

from .parsing_utils import FeatureExtractor

log = logging.getLogger(__file__)

with open(os.path.join(os.path.dirname(__file__), "database_schema2.sql")) as f:
    DB_CREATION_SCRIPT = f.read()

def str_generator(size=6, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

def unique_list(a_list):
    seen = set()
    result = []
    for x in a_list:
        key = repr(x)
        if key not in seen:
            seen.add(key)
            result.append(x)
    return result

class CompilerDatasetLoggingWrapper(CompilerEnvWrapper):
    """A wrapper for an LLVM environment that logs all transitions to an sqlite
    database.

    Wrap an existing LLVM environment and then use it as per normal:

        >>> env = CompilerDatasetLoggingWrapper(
        ...     env=gym.make("llvm-autophase-ic-v0"),
        ...     db_path="example.db",
        ... )

    Connect to the database file you specified:

        $ sqlite3 example.db

    There are two tables:

    (1) States: records every unique combination of benchmark + actions. For
        each entry, records an identifying state ID, the episode reward, and
        whether the episode is terminated:

            sqlite> .mode markdown
            sqlite> .headers on
            sqlite> select * from States limit 5;
            |      benchmark_uri       | done | ir_instruction_count_oz_reward |                 state_id                 |    actions     |
            |--------------------------|------|--------------------------------|------------------------------------------|----------------|
            | generator://csmith-v0/99 | 0    | 0.0                            | d625b874e58f6d357b816e21871297ac5c001cf0 |                |
            | generator://csmith-v0/99 | 0    | 0.0                            | d625b874e58f6d357b816e21871297ac5c001cf0 | 31             |
            | generator://csmith-v0/99 | 0    | 0.0                            | 52f7142ef606d8b1dec2ff3371c7452c8d7b81ea | 31 116         |
            | generator://csmith-v0/99 | 0    | 0.268005818128586              | d8c05bd41b7a6c6157b6a8f0f5093907c7cc7ecf | 31 116 103     |
            | generator://csmith-v0/99 | 0    | 0.288621664047241              | c4d7ecd3807793a0d8bc281104c7f5a8aa4670f9 | 31 116 103 109 |

    (2) Observations: records pickled, compressed, adn text observation values
        for each unique state.
    """

    def __init__(
        self,
        env: LlvmEnv,
        db_path: Path,
        vocab_db_path: Path = None,
        commit_frequency_in_seconds: int = 300,
        max_state_buffer_length: int = 5000,
        online_update_vocab=False,
        graph_version=0,
    ):
        super().__init__(env)
        assert isinstance(self.unwrapped, LlvmEnv), "Requires LlvmEnv base environment"
        db_path.parent.mkdir(exist_ok=True, parents=True)
        self._init_params = dict(
            db_path=db_path,
            vocab_db_path=vocab_db_path,
            commit_frequency_in_seconds=commit_frequency_in_seconds,
            max_state_buffer_length=max_state_buffer_length,
            online_update_vocab=online_update_vocab,
            graph_version=graph_version,
        )
        self.connection = sqlite3.connect(db_path, timeout=3200)
        self.cursor = self.connection.cursor()
        self.commit_frequency = commit_frequency_in_seconds
        self.max_state_buffer_length = max_state_buffer_length
        self.vocab_db_path = vocab_db_path

        self.cursor.executescript(DB_CREATION_SCRIPT)
        self.connection.commit()
        self.last_commit = time()

        self.transition_buffer_current = []
        self.transition_buffer = []
        self.state_buffer = []

        # House keeping notice: Keep these lists in sync with record().
        self._observations = [
            self.env.observation.spaces["IrSha1"],
            self.env.observation.spaces["Ir"],
            self.env.observation.spaces["Programl"],
            self.env.observation.spaces["Autophase"],
            self.env.observation.spaces["InstCount"],
            self.env.observation.spaces["IrInstructionCount"],
        ]
        self._rewards = [
            self.env.reward.spaces["IrInstructionCountOz"],
            self.env.reward.spaces["IrInstructionCount"],
        ]
        self._reward_totals = np.zeros(len(self._rewards))

        # Load vocab
        self.feature_extractor = FeatureExtractor(vocab_db_path, online_update=online_update_vocab, graph_version=graph_version)
        self.new_vocab = set()
        self.traj_id = None

    def __exit__(self, *args, **kwargs):
        self.save_vocabs_encoding()
        super().__exit__(*args, **kwargs)

    def save_vocabs_encoding(self):
        if self.vocab_db_path is not None:
            # Save it to the current database. 
            self.feature_extractor.save_vocab_to_db(self.cursor, table_name="VocabsForEncoding")
            self.connection.commit()

    def flush(self, force=False) -> None:
        """Flush the buffered steps and observations to database."""
        if (not force 
            and len(self.state_buffer) < self.max_state_buffer_length
            and time() - self.last_commit < self.commit_frequency
        ):
            # Not yet need flushing
            return

        n_steps, n_observations, n_vocab = len(self.state_buffer), len(self.transition_buffer), len(self.new_vocab)

        # Nothing to flush.
        if not n_steps:
            return

        with Timer() as flush_time:
            # House keeping notice: Keep these statements in sync with record().
            try:
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO States VALUES (?, ?, ?, ?, ?, ?)",
                    self.state_buffer
                )
            except Exception as e:
                print(f"Error INSERT States: {e}")

            try:
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO Transitions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    self.transition_buffer
                )
            except Exception as e:
                print(f"Error INSERT Transitions: {e}")
            try:
                # Save all vocab as well. 
                self.cursor.executemany(
                    "INSERT OR IGNORE INTO Vocabs VALUES (?)",
                    [ (token,) for token in self.new_vocab ]
                )
            except Exception as e:
                print(f"Error INSERT Vocabs: {e}")

            self.state_buffer = []
            self.transition_buffer = []
            self.new_vocab = set()

            self.connection.commit()

        log.info(
            "Wrote %d state records, %d observations and %d vocab in %s. Last flush %s ago",
            n_steps,
            n_observations,
            n_vocab,
            flush_time,
            humanize.naturaldelta(time() - self.last_commit),
        )
        self.last_commit = time()

    def reset(self, *args, **kwargs):
        self.on_traj_done()

        observation = self.env.reset(*args, **kwargs)
        observations, rewards, done, info = self.env.step(
            action=[], observation_spaces=self._observations, reward_spaces=self._rewards
        )
        assert info['action_had_no_effect'] == True
        assert not done, f"reset() failed! {info}"
        self._reward_totals = np.array(rewards, dtype=np.float32)
        self.traj_id = observations[0] + str_generator(size=5)  # IrSha1 + random str
        self.traj_length = 0
        self.save_traj()

        self.record_state(self.actions, observations, done)
        self.prev_state_id = observations[0]

        return observation

    def save_traj(self):
        self.cursor.execute("INSERT INTO Trajectories VALUES (?,?)", (self.traj_id, self.traj_length))  # throw exception if fail
        self.connection.commit()

    def update_traj(self):
        if self.traj_id is None:
            return
        # logging.info(f"update traj {(self.traj_length, self.traj_id)}")
        # logging.info(f"{self.actions}")
        self.cursor.execute("UPDATE Trajectories SET traj_length = ? WHERE traj_id = ?", (self.traj_length, self.traj_id))
        self.connection.commit()

    def _split(self, mapping, s1, s2):
        r1 = [ mapping[repr(s)] for s in s1 ]
        r2 = [ mapping[repr(s)] for s in s2 ]
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

        overall_obs_spaces = unique_list(self._observations + [self.observation_space_spec]) 
        overall_reward_spaces = unique_list(self._rewards + [self.reward_space])
        
        observations, rewards, done, info = self.env.step(
            action=action, 
            observation_spaces=overall_obs_spaces, 
            reward_spaces=overall_reward_spaces, 
            **kwargs2,
        )

        observation_mapping = { repr(k): v for k, v in zip(overall_obs_spaces, observations) } 
        reward_mapping = { repr(k) : v for k, v in zip(overall_reward_spaces, rewards) } 

        external_obs_space = kwargs.get("observation_spaces", [self.observation_space_spec])
        external_reward_space = kwargs.get("reward_spaces", [self.reward_space])  # checked, it is [IrInstructionCountOz]

        # Properly handling observation/reward spaces. 
        internal_obs, external_obs = self._split(observation_mapping, self._observations, external_obs_space)
        internal_rewards, external_rewards = self._split(reward_mapping, self._rewards, external_reward_space)

        self._reward_totals += internal_rewards

        self.record_state(self.actions, internal_obs, done)
        state_id = internal_obs[0]
        # assert isinstance(action, int), f"action is {action}"
        if isinstance(action, int):
            # only start recording after the first empty action '[]'
            self.record_transition(self.prev_state_id, action, state_id, done, internal_obs, internal_rewards, info)
        self.prev_state_id = state_id

        if len(external_obs) == 1:
            external_obs = external_obs[0]

        if len(external_rewards) == 1:
            external_rewards = external_rewards[0] 

        return external_obs, external_rewards, done, info

    def record_state(self, actions, observations, done) -> None:
        state_id, ir, programl, _, _, IrInstructionCount = observations

        # TODO: figure out if some graphs have no nodes (when ctrl+c)
        # It seems that ctrl+c will interupt the generation of the graph in the compiler backend
        # and lead to the backend returning an empty graph
        if programl.number_of_nodes() == 0:
            return

        for node_name, text in programl.nodes(data="text"):
            # This is basically a networkx thing, where 0th index is
            # the node index and then the last index is always the text feature
            self.new_vocab.add(text)

        dgl_graph = self.feature_extractor.process_nx_graph(programl)

        self.state_buffer.append(
            (
                str(self.benchmark.uri),
                state_id,
                zlib.compress(pickle.dumps(dgl_graph)),
                # pickle.dumps(dgl_graph),
                1 if done else 0,
                # Action history
                " ".join(str(x) for x in actions),
                int(IrInstructionCount),
            )
        )

    def record_transition(self, state_id, action, next_state_id, done, observations, rewards, info) -> None:
        _, _, _, autophase, instcount, instruction_count = observations
        # Current reward.
        instruction_count_reward = float(rewards[0])
        self.traj_length += 1

        self.transition_buffer_current.append(
            dict(
                state_id=state_id,
                traj_id=self.traj_id,
                traj_step=self.traj_length,
                action=action,
                next_state_id=next_state_id,
                done=done,
                instruction_count=instruction_count,
                ir_instruction_count_reward=int(rewards[1]),  # int, ir_instruction_count reduction
                instruction_count_reward=instruction_count_reward,
                cumulative_reward=0, # cumulative reward, currently set to zero. 
                cumulative_reward2=0, # cumulative reward for ir_instruction_count, currently set to zero. 
                action_had_no_effect=int(info["action_had_no_effect"]),
                autophase=" ".join(str(x) for x in autophase),
                instcount=" ".join(str(x) for x in instcount),
            )
        )

        if done:
            self.on_traj_done()

    def on_traj_done(self):
        traj = self.transition_buffer_current 
        n = len(traj)
        last_n = 5

        # If the trajectory ends when done = True, then we save all trajectory, 
        # Else we cut the last part (since their cumulative rewards are not accurate)
        if n == 0 or (not traj[-1]["done"] and n <= last_n):
            self.transition_buffer_current = []
            return

        # Then we compute the cumulative rewards. 
        # 1. Immediate reward is metric(s_t) - metric(s_{t-1}) 
        # 2. Cumulative reward is weighted sum of immediate reward.  
        rewards = [traj[i]["instruction_count_reward"] for i in range(n)]
        # diff = rewards[1:] - rewards[:-1] 
        # Current reward is already code delta (size of code changes before/after action)
        cum_rewards = [0] * n # np.zeros((n))
        gamma = 0.9
        cum_reward = 0
        for i in range(n - 1, -1, -1):
            cum_rewards[i] = cum_reward = cum_reward * gamma + rewards[i]

        # Write them back. 
        for i in range(n):
            traj[i]["cumulative_reward"] = cum_rewards[i] 

        # computing rewards of instruction count reducion
        rewards = [traj[i]["ir_instruction_count_reward"] for i in range(n)]
        cum_rewards = [0] * n
        gamma = 0.9
        cum_reward = 0
        for i in range(n - 1, -1, -1):
            cum_rewards[i] = cum_reward = cum_reward * gamma + rewards[i]

        # Write them back. 
        for i in range(n):
            traj[i]["cumulative_reward2"] = cum_rewards[i] 

        # an order list of keys
        keys = [
            "state_id", "traj_id", "traj_step", "action", "next_state_id", "done", 
            "instruction_count", "ir_instruction_count_reward", "instruction_count_reward", 
            "cumulative_reward", "cumulative_reward2", "action_had_no_effect", "autophase", "instcount"
        ]

        for i in range(n - last_n):
            self.transition_buffer.append(tuple(traj[i][k] for k in keys))
        self.transition_buffer_current = []
        self.update_traj()
        self.flush()

    def close(self):
        self.on_traj_done()
        self.flush(force=True)
        self.env.close()

    def fork(self):
        fork_env = type(self)(env=self.env.fork(), **self._init_params)
        # FIXME: copy other attributes to forked env?
        # TODO: share db connection?
        # fork_env.connection.close()
        # fork_env.connection = self.connection
        # fork_env.cursor = self.cursor
        return fork_env


def merge_dbs(db_paths: List[str], out_path: str, tables : List[str] = list()):
    log.info("Merging %d databases to %s", len(db_paths), out_path)

    db_exist = os.path.isfile(out_path)
    connection = sqlite3.connect(out_path)
    cursor = connection.cursor()

    if not db_exist:
        log.info(f"Creating new dataset file {out_path}")
        cursor.executescript(DB_CREATION_SCRIPT)
        connection.commit()

    if len(tables) == 0:
        tables = ["States", "Transitions", "Vocabs"]

    assert len(db_paths) > 0, "Need at least one database file!"

    for db_path in db_paths:
        log.info(f"Merging {db_path}")
        cursor.execute("ATTACH DATABASE ? AS other", (db_path,))

        for table in tables:
            cursor.execute(f"INSERT OR IGNORE INTO {table} SELECT * from other.{table}")

        connection.commit()
        cursor.execute("DETACH DATABASE other")

    cnts = { table : cursor.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in tables }
    connection.commit()

    s = f"Merged database: " + " ".join([ f"{humanize.intcomma(cnt)} in {table} " for table, cnt in cnts.items()])
    log.info(s)
    connection.close()


# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS States (
    benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
    state_id TEXT NOT NULL UNIQUE,       -- 40-char sha1
    graph BLOB,                          -- pyg graph, in pickle format, need to use zlib.decompress()
    autophase TEXT,                      -- Decode: np.array([int(x) for x in field.split()], dtype=np.int64)
    IrInstructionCount INTEGER NOT NULL, -- raw ir_instruction_count of the this state
    num_nodes INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS Transitions (
    benchmark_uri TEXT NOT NULL,
    state_id TEXT NOT NULL,
    action_value INTEGER NOT NULL,     -- the action performed
    reward REAL NOT NULL,              -- the reward received immediately after the action
    cumulative_reward REAL NOT NULL,   -- discounted cumulative reward
    next_state_id TEXT NOT NULL,       -- the observed state_id after applying the action
    num_nodes INTEGER NOT NULL,
    traj_id TEXT NOT NULL,
    traj_step INTEGER NOT NULL,
    advantage REAL,                    -- the advantage associated with the state-action
    logp REAL,                         -- log probability of selecting the action_value
    time_stamp REAL                    -- the time stamp of the creation of this transition
);

CREATE TABLE IF NOT EXISTS Trajectories (
    traj_id TEXT NOT NULL UNIQUE,      -- 45-char: starting state's sha1 + 5 random char
    benchmark_uri TEXT NOT NULL,
    state_ids TEXT NOT NULL,           -- the sequence of all observed state id [n+1]
    actions TEXT NOT NULL,             -- the sequence of actions. Decode: [int(x) for x in field.split()] [n]
    rewards BLOB NOT NULL,             -- the rewards (IrInstructionCount / ir_current) received immediately after each action [n]
    graph_repr BLOB                    -- the representation of each observed state [n+1]
);

CREATE TABLE IF NOT EXISTS TrainerProgress (
    read_rows INTEGER NOT NULL,        -- the number of rows the trainer has read
    num_rows_per_iter REAL NOT NULL    -- an estimate of number of rows read per trainer iteration
);

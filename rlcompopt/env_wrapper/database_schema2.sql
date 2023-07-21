
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS States (
  benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
  state_id TEXT NOT NULL UNIQUE,              -- 40-char sha1.
  dgl_graph BLOB NOT NULL,             -- dgl graph, in pickle format, need to use zlib.decompress()
  done INTEGER NOT NULL,               -- 0 = False, 1 = True.
  actions TEXT NOT NULL,               -- Decode: [int(x) for x in field.split()]
  IrInstructionCount INTEGER NOT NULL, 
  PRIMARY KEY (benchmark_uri, actions),
  FOREIGN KEY (state_id) REFERENCES Transitions(state_id) ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS Transitions (
    state_id TEXT NOT NULL,            -- 40-char sha1.
    traj_id TEXT NOT NULL,             -- 45-char: starting state's sha1 + 5 random char
    traj_step INTEGER,                 -- length of the traj, == total transition pairs (this number could be larger if the end of traj is cut off)
    action_value TEXT NOT NULL,
    next_state_id TEXT NOT NULL,            -- 40-char sha1.
    done INTEGER NOT NULL,               -- 0 = False, 1 = True.
    ir_instruction_count INTEGER NOT NULL,  -- raw ir_instruction_count of the next state
    ir_instruction_count_reward INTEGER NOT NULL,   -- next state's ir_instruction_count reduction from this state
    ir_instruction_count_oz_reward REAL NULLABLE,
    cumulative_reward REAL NULLABLE,
    cumulative_reward2 REAL NULLABLE,               -- cumulative ir_instruction_count_reward
    action_had_no_effect INTEGER NOT NULL,   -- 0 = False, 1 = True. Note: there maybe a bug in compiler_gym: action_had_no_effect can be 0 (meaning the action has effect) even though state_id and next_state_id are the same
    autophase TEXT NOT NULL,                    -- Decode: np.array([int(x) for x in field.split()], dtype=np.int64)
    instcount TEXT NOT NULL,                    -- Decode: np.array([int(x) for x in field.split()], dtype=np.int64)
    PRIMARY KEY (traj_id, traj_step)
);

CREATE TABLE IF NOT EXISTS Trajectories (
    traj_id TEXT NOT NULL UNIQUE, 
    traj_length INTEGER,
    PRIMARY KEY (traj_id)
);

CREATE TABLE IF NOT EXISTS Vocabs (
    token TEXT NOT NULL UNIQUE, 
    PRIMARY KEY (token)
);

-- The vocabulary used for encoding graphs.
CREATE TABLE IF NOT EXISTS VocabsForEncoding (
    token TEXT NOT NULL UNIQUE, 
    PRIMARY KEY (token)
);

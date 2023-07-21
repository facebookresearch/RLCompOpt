
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS Metrics (
    benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
    score REAL,  -- (Oz - min_ir) / Oz
    diff REAL,  -- Oz - min_ir
    oz REAL, -- Oz
    o0 REAL, -- O0
    best_ir_seq TEXT, -- the ir count seq corresponding to the min_ir; space separated
    best_action_seq TEXT -- the action seq corresponding to the min_ir; space separated
);

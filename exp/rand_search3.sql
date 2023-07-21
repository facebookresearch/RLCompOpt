
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS SearchScores (
    benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
    score1 REAL,  -- (Oz - min_ir) / Oz
    score2 REAL,  -- (Oz - min_ir) / Oz
    score3 REAL,  -- (Oz - min_ir) / Oz
    score4 REAL,  -- (Oz - min_ir) / Oz
    score5 REAL  -- (Oz - min_ir) / Oz
);

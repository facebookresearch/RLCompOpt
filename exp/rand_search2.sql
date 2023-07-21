
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS Val (
    benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
    oz REAL, -- Oz
    o0 REAL, -- O0
    val_seq BLOB  -- the cumulative rewards of each seq; python list
);

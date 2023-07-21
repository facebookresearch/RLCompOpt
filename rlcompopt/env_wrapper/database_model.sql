
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS Models (
  config BLOB,                  -- config for initializing the model with hydra
  kwargs BLOB,                  -- kwargs for initializing the model
  state_dict BLOB NOT NULL,     -- the state dict
  ema_state_dict BLOB           -- the state dict of the ema model
);

CREATE TABLE IF NOT EXISTS Performance (
  model_id INTEGER NOT NULL,    -- the rowid of the model being evaluated
  split_tag TEXT NOT NULL,      -- dataset split
  total_metric REAL NOT NULL,   -- total percent improvement over Oz
  mean_metric REAL NOT NULL,    -- average percent improvement over Oz
  table_rows BLOB NOT NULL,     -- the table rows that can be put into tabulate.tabulate
  table_str TEXT NOT NULL       -- the str of tabulate.tabulate(rows)
);

CREATE TABLE IF NOT EXISTS ValBest (
  model_id INTEGER NOT NULL    -- the rowid of the model with smallest val loss
);

CREATE TABLE IF NOT EXISTS Signal (
  done INTEGER
);

# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CREATE TABLE IF NOT EXISTS Socket (
  ip TEXT NOT NULL,         -- the ip address of the socket
  port INTEGER NOT NULL     -- the port number
);

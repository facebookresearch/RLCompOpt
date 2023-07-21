
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class ReprQueue:
    """
    A queue holding the representations from the EMA encoder, as in MoCo:
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, dim=64, K=1000, min_K=50):

        self.K = K  # number of trajectories to store in buffer
        self.min_K = min_K  # minimum number of trajectorie to enable training

        self.buffer = torch.zeros(0, dim)
        self.traj_batch_idx = [0]  # storing the starting and ending idx of each chunk of trajectory representations, len == num_traj + 1
        self.traj_ids = []  # a list that maps traj_id to an idx, where traj_batch_idx[idx] = the starting index of the traj buffer in self.buffer

    @torch.no_grad()
    def collate_fn(self, trajs: List[Tuple[str, int, int, torch.Tensor]]):
        """
        Args:
            trajs: a list of [traj_id, current_repr_idx, next_repr_idx, repr]
        """

        # remove some traj buffer if the new buffer is going to be too long
        this_traj_ids = set([tid for tid, *_ in trajs] + self.traj_ids)
        num_rm = max(0, len(this_traj_ids) - self.K)
        if num_rm > 0:
            new_start = self.traj_batch_idx[num_rm]
            traj_batch_idx = np.array(self.traj_batch_idx[num_rm:], dtype=np.int32)
            self.traj_batch_idx = (traj_batch_idx - new_start).tolist()  # traj_batch_idx starts from 0
            self.buffer = self.buffer[new_start:]
            self.traj_ids = self.traj_ids[num_rm:]

        curr = []  # holding the idx of the current repr in the buffer
        next_ = []  # holding the idx of the next repr in the buffer

        new_buffer = [self.buffer]
        new_idx = self.traj_batch_idx

        for traj_id, current_repr_idx, next_repr_idx, repr_ in trajs:
            if traj_id not in self.traj_ids:
                bz = repr_.shape[0]
                starting_idx = new_idx[-1]
                new_idx.append(starting_idx + bz)
                new_buffer.append(repr_)
                self.traj_ids.append(traj_id)
            else:
                idx = self.traj_ids.index(traj_id)
                starting_idx = new_idx[idx]
            # starting_idx of the current buffer
            curr.append(current_repr_idx + starting_idx)
            next_.append(next_repr_idx + starting_idx)

        self.buffer = torch.cat(new_buffer, dim=0)
        if len(self.traj_ids) >= self.min_K:
            current_state_idx = torch.tensor(curr, dtype=torch.long)
            next_state_idx = torch.tensor(next_, dtype=torch.long)
            return self.buffer.clone(), current_state_idx, next_state_idx
        return None

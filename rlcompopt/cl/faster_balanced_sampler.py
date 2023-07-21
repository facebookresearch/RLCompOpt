
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DistributedSampler

log = logging.getLogger(__file__)


class BinPacking:
    def __init__(
        self,
        items: List[Tuple[int, float]],
        bin_size: float,
        full_rate: float = 0.9,
        rank: int = 0,
        num_replicas: int = 1,
    ) -> None:
        """
        BinPacking puts the items of various weight in bins without exceeding the bin size.
        It yields the full bin immediately when it is ready.
        The full bins are distributed to `num_replicas` processes and each process has its own set of bins.
        The computation is basically the same for each process to avoid process communication, but each process
        only outputs the bins that belong to itself.
        """
        self.rank = rank
        self.num_replicas = num_replicas
        self.bin_size = bin_size
        self.full_size = self.bin_size * full_rate
        self.items = items
        self.bins = []
        self.loaded_bins = []
        self.loaded_weight = []  # this is for recording total weight in self.bins
        self.next_bin_idx = rank  # only outputs the bin for this process

    def __iter__(self):
        for idx, weight in self.items:
            found = False
            for i, bin_ in enumerate(self.bins):
                tmp = self.loaded_weight[i] + weight
                if tmp <= self.bin_size:
                    # fits into this bin, add it to bin
                    bin_.append(idx)
                    self.loaded_weight[i] = tmp
                    found = True
                    if tmp >= self.full_size:
                        # this bin reachs its full size, so move it to the list of loaded bins and yield it
                        self.loaded_bins.append(bin_)
                        if self.next_bin_idx < len(self.loaded_bins):
                            yield self.loaded_bins[self.next_bin_idx]
                            self.next_bin_idx += self.num_replicas
                        del self.bins[i]
                        del self.loaded_weight[i]
                    break
            if not found:
                # does not fit to any bins, put it to a new bin
                self.bins.append([idx])
                self.loaded_weight.append(weight)
        num_bins = len(self.loaded_bins) // self.num_replicas
        num_last_loaded_bins = len(self.loaded_bins) - num_bins * self.num_replicas
        # check if the number of batches is divisible by the number of processes
        if len(self.loaded_bins) == 0 or (
            num_last_loaded_bins > 0 and self.rank >= num_last_loaded_bins
        ):
            # this process has not yet output its own last batch
            padding_size = self.num_replicas - num_last_loaded_bins - len(self.bins)
            if padding_size > 0:
                # pad bins to self.bins
                pad_batches = (
                    self.loaded_bins + self.bins
                )  # prioritize padding with full bins
                if padding_size <= len(pad_batches):
                    self.bins += pad_batches[:padding_size]
                else:
                    self.bins += (
                        pad_batches * math.ceil(padding_size / len(pad_batches))
                    )[:padding_size]
            self.loaded_bins += self.bins[: self.num_replicas - num_last_loaded_bins]
            yield self.loaded_bins[self.next_bin_idx]


class BalancedBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        max_size: int,
        full_rate: float,
        size_func: Callable,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.max_size = max_size
        self.full_rate = full_rate
        self.datasize = [
            size_func(d) for d in self.dataset if size_func(d) <= self.max_size
        ]
        assert len(self.datasize) == len(
            self.dataset
        ), "Some items have size exceeding the max_size"

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        datasize = [(idx, self.datasize[idx]) for idx in indices]
        bin_pack = BinPacking(
            datasize, self.max_size, self.full_rate, self.rank, self.num_replicas
        )

        return bin_pack.__iter__()

    def __len__(self):
        # The length of BalancedBatchSampler is undefined as it could change
        if not hasattr(self, "_len_batches"):
            self._len_batches = 0
            idx = []
            for b in self.__iter__():
                self._len_batches += 1
                idx.extend(b)
            idx = set(idx)
            num_covered = len(idx & set(range(len(self.dataset))))
            log.info(
                f"this iteration in a single process covers {num_covered}/{len(self.dataset)} ({num_covered/len(self.dataset):.2%}) datapoints"
            )
        return self._len_batches

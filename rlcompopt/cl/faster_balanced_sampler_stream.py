
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import sqlite3
import time
from typing import Optional, Iterable, Union

import GPUtil
import torch
import torch.distributed as dist
from rlcompopt.utils import get_rank

log = logging.getLogger(__file__)


def parse_benchmark(bm : str):
    typ, _, rest = bm.partition('://')
    assert _ == '://'
    dataset, _, name = rest.partition('/')
    assert _ == '/'
    return typ, dataset, name


class DatabaseStream:
    def __init__(
        self,
        db_path: str,
        db_fields: str,
        db_table: str = "Transitions",
        start: int = 0,
        sleep: int = 0.1,
        circulate_data=False,
        num_records: int = 60000,
        eval_data_len: int = 0,  # if load eval data, set a positive number
        exclude_sets: str = None,
        seq_classification: bool = False,  # in seq_classification, we just need the initial state
    ) -> None:
        self.db_path = db_path
        self.db_fields = db_fields
        self.db_table = db_table
        self.start = start
        self.sleep = sleep
        self.connection = None
        self.cursor = None
        self.circulate_data = circulate_data
        self.num_records = num_records
        self.eval_data_len = eval_data_len
        self.seq_classification = seq_classification
        self.__len = 0

        self.exclude_sets = exclude_sets
        if self.exclude_sets is not None:
            self.exclude_sets = exclude_sets.split("_")
        self.epoch = 0

    def _setup(self):
        self.connection = sqlite3.connect(self.db_path, timeout=1200)
        self.cursor = self.connection.cursor()
        if self.circulate_data:
            return

        # resume from the last training rows
        rec = list(
            self.cursor.execute(
                "SELECT read_rows FROM TrainerProgress ORDER BY rowid DESC LIMIT 1"
            )
        )
        if rec:
            self.start = rec[0][0]
            print(f"Training resume from database row {self.start}")

    def __len__(self):
        return self.__len

    def is_excluded(self, b: str):
        for dataset in self.exclude_sets:
            if b.find(dataset) >= 0:
                return True
        return False

    def __iter__(self):
        while True:
            try:
                time.sleep(2)  # wait for the generator to create the database
                self._setup()
                break
            except sqlite3.OperationalError:
                print(f"Streaming database not ready. Retrying...")
                continue
        if self.seq_classification:
            additional = "AND traj_step = 0 "
        else:
            additional = ""

        # streaming of rowid and num_nodes
        idx = self.start

        if self.circulate_data:
            new_rows = list(
                self.cursor.execute(
                    f"SELECT rowid, {self.db_fields}, benchmark_uri from {self.db_table} WHERE rowid > 0 AND rowid < {self.num_records} {additional}ORDER BY rowid"
                )
            )
            if self.exclude_sets is not None:
                new_rows = [ro for ro in new_rows if not self.is_excluded(ro[2])]

            if self.eval_data_len > 0:
                benchmark_uri = set(record[2] for record in new_rows)
                new_rows = list(
                    self.cursor.execute(
                        f"SELECT rowid, {self.db_fields}, benchmark_uri from {self.db_table} WHERE rowid > {self.num_records} {additional}ORDER BY rowid"
                    )
                )
                if self.exclude_sets is not None:
                    new_rows = [ro for ro in new_rows if not self.is_excluded(ro[2])]
                # remove those rows with same benchmark_uri in any of the training data
                new_rows = [
                    record for record in new_rows if record[2] not in benchmark_uri
                ]
                new_rows = new_rows[: self.eval_data_len]
            self.__len = len(new_rows)
            new_rows = [record[:2] for record in new_rows]

        while True:
            try:
                if self.circulate_data:
                    # shuffle
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    indices = torch.randperm(len(new_rows), generator=g).tolist()
                    for i_ in indices:
                        yield new_rows[i_]
                    self.epoch += 1
                    continue

                # below is for loading online data stream
                rec = None
                new_rows = list(
                    self.cursor.execute(
                        f"SELECT rowid, {self.db_fields} from {self.db_table} WHERE rowid > {idx} ORDER BY rowid"
                    )
                )
                for rec in new_rows:
                    yield rec
                # reached the end of records, so wait for new records
                # time.sleep(self.sleep)
                if rec is not None:
                    idx = rec[0]
            except sqlite3.OperationalError as e:
                print(f"OperationalError: {e}, skipping...")
                log.info(f"OperationalError: {e}, skipping...")
                time.sleep(self.sleep)
            except (sqlite3.DatabaseError, sqlite3.DataError) as e:
                print(f"DatabaseError: {e}. Reconnecting to database...")
                log.info(f"DatabaseError: {e}. Reconnecting to database...")
                time.sleep(self.sleep)
                self.connection.close()
                self.connection = sqlite3.connect(self.db_path, timeout=1200)
                self.cursor = self.connection.cursor()


class WeightedActionDatabaseStream:
    def __init__(
        self,
        db_path: str,
        db_fields: str,
        db_table: str = "Transitions",
        start: int = 0,
        sleep: int = 0.1,
        circulate_data=False,
        num_records: int = 60000,
        eval_data_len: int = 0,  # if load eval data, set a positive number
        exclude_sets: str = None,
        seq_classification: bool = False,  # in seq_classification, we just need the initial state
    ) -> None:
        self.db_path = db_path
        self.db_fields = db_fields
        self.db_table = db_table
        self.start = start
        self.sleep = sleep
        self.connection = None
        self.cursor = None
        self.circulate_data = circulate_data
        self.num_records = num_records
        self.eval_data_len = eval_data_len
        self.seq_classification = seq_classification
        self.__len = 0

        self.exclude_sets = exclude_sets
        if self.exclude_sets is not None:
            self.exclude_sets = exclude_sets.split("_")
        self.epoch = 0

    def is_excluded(self, b: str):
        for dataset in self.exclude_sets:
            if b.find(dataset) >= 0:
                return True
        return False

    def _setup(self):
        self.connection = sqlite3.connect(self.db_path, timeout=1200)
        self.cursor = self.connection.cursor()

        if self.circulate_data:

            new_rows = list(
                self.cursor.execute(
                    f"SELECT rowid, {self.db_fields}, benchmark_uri from {self.db_table} WHERE rowid > 0 AND rowid < {self.num_records} ORDER BY rowid"
                )
            )
            if self.exclude_sets is not None:
                new_rows = [ro for ro in new_rows if not self.is_excluded(ro[2])]

            if self.eval_data_len > 0:
                benchmark_uri = set(record[2] for record in new_rows)
                new_rows = list(
                    self.cursor.execute(
                        f"SELECT rowid, {self.db_fields}, benchmark_uri from {self.db_table} WHERE rowid > {self.num_records} ORDER BY rowid"
                    )
                )
                if self.exclude_sets is not None:
                    new_rows = [ro for ro in new_rows if not self.is_excluded(ro[2])]
                # remove those rows with same benchmark_uri in any of the training data
                new_rows = [
                    record for record in new_rows if record[2] not in benchmark_uri
                ]
                new_rows = new_rows[: self.eval_data_len]
            self.__len = len(new_rows)
            actions = [parse_benchmark(record[2])[1] for record in new_rows]
            datasets = {bm : i for i, bm in enumerate(set(actions))}
            actions = [datasets[bm] for bm in actions]
            actions = torch.tensor(actions, dtype=torch.long)
            a_, counts = actions.unique(return_counts=True)
            a_counts = [0] * len(datasets)
            for a, c in zip(a_.tolist(), counts.tolist()):
                a_counts[a] = c
            weights = [1 / c if c != 0 else 0. for c in a_counts]
            print(f"action counts: {a_counts} \naction weight: {weights}")
            weights = torch.tensor(weights, dtype=torch.float)
            self.resampling_weight = weights[actions]
            self.new_rows = [record[:2] for record in new_rows]

    def __len__(self):
        return self.__len

    def __iter__(self):
        while True:
            try:
                time.sleep(2)  # wait for the generator to create the database
                self._setup()
                break
            except sqlite3.OperationalError:
                print(f"Streaming database not ready. Retrying...")
                continue

        # streaming of rowid and num_nodes
        idx = self.start
        resampling_weight = self.resampling_weight
        new_rows = self.new_rows
        len_data = resampling_weight.numel()

        while True:
            try:
                if self.circulate_data:
                    # shuffle
                    g = torch.Generator()
                    g.manual_seed(self.epoch)
                    indices = torch.multinomial(resampling_weight, len_data, replacement=True, generator=g).tolist()
                    for i_ in indices:
                        yield new_rows[i_]
                    self.epoch += 1
                    continue

                # below is for loading online data stream
                rec = None
                new_rows = list(
                    self.cursor.execute(
                        f"SELECT rowid, {self.db_fields} from {self.db_table} WHERE rowid > {idx} ORDER BY rowid"
                    )
                )
                for rec in new_rows:
                    yield rec

                if rec is not None:
                    idx = rec[0]
            except sqlite3.OperationalError as e:
                print(f"OperationalError: {e}, skipping...")
                log.info(f"OperationalError: {e}, skipping...")
                time.sleep(self.sleep)
            except (sqlite3.DatabaseError, sqlite3.DataError) as e:
                print(f"DatabaseError: {e}. Reconnecting to database...")
                log.info(f"DatabaseError: {e}. Reconnecting to database...")
                time.sleep(self.sleep)
                self.connection.close()
                self.connection = sqlite3.connect(self.db_path, timeout=1200)
                self.cursor = self.connection.cursor()


def get_gpu_avail():
    deviceIDs = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])
    if deviceIDs:
        return
    gpus = GPUtil.getGPUs()
    util = [gpu.memoryUtil for gpu in gpus]  # float(memoryUsed)/float(memoryTotal)
    if util:
        # max_util = max(util)
        rank = get_rank()
        if rank >= len(util):
            return
        r = 1. / util[rank]
        if r > 1.03:
            return 1.01
        elif r < 1.02:
            return 0.99


class BinPacking:
    def __init__(
        self,
        idx_streamer,
        bin_size: float,
        full_rate: float = 0.9,
        rank: int = 0,
        num_replicas: int = 1,
        auto_batchsize: bool = False,
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
        self.idx_streamer = idx_streamer
        self.bins = []
        self.loaded_bins = []
        self.loaded_weight = []  # this is for recording total weight in self.bins
        self.next_bin_idx = rank  # only outputs the bin for this process
        self.auto_batchsize = auto_batchsize

    def __iter__(self):
        for idx, weight in self.idx_streamer:
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
                            if self.auto_batchsize:
                                rate = get_gpu_avail()
                                if rate is not None:
                                    self.full_size *= rate
                            yield self.loaded_bins[self.next_bin_idx]
                            del self.loaded_bins[
                                : self.next_bin_idx
                            ]  # delete the loaded bins to save memory
                            self.next_bin_idx = 0
                            self.next_bin_idx += self.num_replicas
                        # loaded_weight is associated with bins
                        del self.bins[i]
                        del self.loaded_weight[i]
                    break
            if not found:
                # does not fit to any bins, put it to a new bin
                self.bins.append([idx])
                self.loaded_weight.append(weight)


class BalancedStreamer:
    def __init__(
        self,
        db_path: str,
        max_size: int,
        full_rate: float,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        circulate_data=False,
        eval_data_len=0,
        num_records=10000,  # for offline training
        exclude_sets=None,
        weight_data_resample=False,
        seq_classification=False,
        auto_batchsize=False,
    ) -> None:
        self.db_path = db_path
        self.max_size = max_size
        self.full_rate = full_rate
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.circulate_data = circulate_data
        self.eval_data_len = eval_data_len
        self.num_records = num_records
        self.exclude_sets = exclude_sets
        self.weight_data_resample = weight_data_resample
        self.seq_classification = seq_classification
        self.auto_batchsize = auto_batchsize
        self.__len = float("inf")  # a streaming dataset has infinite length

    def __iter__(self):
        # shuffling is handled by the source of streaming data

        db_table = "Transitions"
        if self.weight_data_resample:
            Database = WeightedActionDatabaseStream
        else:
            Database = DatabaseStream
        db_streamer = Database(
            self.db_path,
            "num_nodes",
            db_table=db_table,
            circulate_data=self.circulate_data,
            eval_data_len=self.eval_data_len,
            num_records=self.num_records,
            exclude_sets=self.exclude_sets,
            seq_classification=self.seq_classification,
        )
        self.db_streamer = db_streamer
        bin_pack = BinPacking(
            db_streamer, self.max_size, self.full_rate, self.rank, self.num_replicas, self.auto_batchsize
        )

        return bin_pack.__iter__()

    def __len__(self):
        return len(self.db_streamer) or self.__len


class SocketDataPacking:
    def __init__(self, bin_size: float, full_rate: float = 0.9, num_max_bin=10):
        """
        Similar to the `BinPacking` class
        """
        self.bin_size = bin_size
        self.num_max_bin = num_max_bin
        self.full_size = self.bin_size * full_rate
        self.bins = []
        self.loaded_weight = []  # this is for recording total weight in self.bins

    def pack(self, idx, weight):
        found = False
        for i, bin_ in enumerate(self.bins):
            tmp = self.loaded_weight[i] + weight
            if tmp <= self.bin_size:
                # fits into this bin, add it to bin
                found = True
                bin_.append(idx)
                self.loaded_weight[i] = tmp
                if tmp >= self.full_size:
                    # this bin reachs its full size, so return it
                    del self.bins[i]
                    del self.loaded_weight[i]
                    return bin_
                break
        if (not found) and weight >= self.full_size:
            return [idx]
        if not found:
            # does not fit to any bins, put it to a new bin
            self.bins.append([idx])
            self.loaded_weight.append(weight)

        # return the largest bin if there are too many bins
        if len(self.bins) > self.num_max_bin:
            bin_weight = torch.tensor(self.loaded_weight, dtype=torch.float)
            idx = torch.max(bin_weight, dim=0)[1].item()
            max_bin = self.bins[idx]
            del self.bins[idx]
            del self.loaded_weight[idx]
            return max_bin

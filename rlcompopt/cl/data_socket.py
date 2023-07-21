
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import socket
import sqlite3
from itertools import islice
from time import sleep
from typing import Dict, List

import zmq

with open(os.path.join(os.path.dirname(__file__), "database_socket.sql")) as f:
    DB_CREATION_SCRIPT = f.read()


class Server:
    def __init__(self) -> None:
        self.conn: zmq.Socket = None
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        port = socket.bind_to_random_port("tcp://*")
        self.ip = get_ip()
        self.addr = (self.ip, port)
        self.conn = socket
        self.setup_db()

    def setup_db(self):
        db_connection = sqlite3.connect(self.socket_db, timeout=1200)
        cursor = db_connection.cursor()
        cursor.executescript(DB_CREATION_SCRIPT)
        db_connection.commit()
        cursor.execute("INSERT INTO Socket VALUES (?, ?)", self.addr)
        db_connection.commit()


class DataServer(Server):
    def __init__(
        self,
        socket_db: str,
    ):
        self.socket_db = socket_db
        super().__init__()
        self.num_sent = 0
        self.num_recv = 0

    def send_pyobj(self, *args, **kwargs):
        self.conn.send_pyobj(*args, **kwargs)
        self.num_sent += 1
        # return self.check_receipt()

    def _check_receipt(self):
        try:
            self.num_recv = int(self.conn.recv_string(flags=zmq.NOBLOCK))
        except Exception:
            pass
        return self.num_sent - self.num_recv


class DataClient:
    def __init__(
        self,
        socket_db: str,
        client_rank: int,
        num_servers: int,
        num_clients: int,
    ) -> None:
        assert num_servers >= num_clients
        self.socket_db = socket_db
        self.addr = []
        self.num_conn = num_servers // num_clients
        assert self.num_conn * num_clients == num_servers
        self.client_rank = client_rank
        self.num_clients = num_clients
        context = zmq.Context()
        self.conn = context.socket(zmq.SUB)
        # self.conn.setsockopt(zmq.CONFLATE, 1)  # only keep the latest message; must before the connect
        self.setup_db()
        self.conn.setsockopt(zmq.SUBSCRIBE, b"")

    def setup_db(self):
        db_connection = sqlite3.connect(self.socket_db, timeout=1200)
        cursor = db_connection.cursor()
        cursor.executescript(DB_CREATION_SCRIPT)
        db_connection.commit()
        for i in range(self.num_conn):
            rowid = self.client_rank + self.num_clients * i + 1
            while True:
                rec = cursor.execute(
                    "SELECT ip, port FROM Socket WHERE rowid = ?", (rowid,)
                )
                result = rec.fetchone()
                if result is not None:
                    ip, port = result
                    self.conn.connect(f"tcp://{ip}:{port}")
                    break
                print("Socket database not yet ready. Waiting...")
                sleep(5)
            print(f"Connects to tcp://{ip}:{port} at socket #{rowid}")

    def serve_data(self):
        # serve data on demand
        # more data are kept in the socket buffer
        try:
            result = self.conn.recv_pyobj(zmq.NOBLOCK)
        except zmq.ZMQError:
            return
        return result


class DataBuffer:
    """A buffer that stores the received data."""

    def __init__(self, key="Transitions", size_idx=6, max_len=20000) -> None:
        self.key = key
        self.size_idx = size_idx
        self.max_len = max_len
        self.buffer = {}
        self.key_id = []

    def store(self, data: Dict[str, Dict]):
        curr_keys = self.buffer.keys()
        for k, v in data.items():
            if k == self.key:
                self.key_id.extend(list(v.keys()))
            if k not in curr_keys:
                self.buffer[k] = v
            else:
                self.buffer[k].update(v)

        # delete oldest buffer
        if len(self.buffer["States"]) > self.max_len:
            first = list(islice(self.buffer["States"], len(self.buffer["States"]) - self.max_len))
            for ss in first:
                self.buffer["States"].pop(ss)
        if len(self.buffer["Transitions"]) > self.max_len:
            first = list(islice(self.buffer["Transitions"], len(self.buffer["Transitions"]) - self.max_len))
            for ss in first:
                self.buffer["Transitions"].pop(ss)
                if ss in self.key_id:
                    self.key_id.remove(ss)
        if len(self.buffer["Trajectories"]) > self.max_len // 10:
            first = list(islice(self.buffer["Trajectories"], len(self.buffer["Trajectories"]) - self.max_len // 10))
            for ss in first:
                self.buffer["Trajectories"].pop(ss)

    def pop(self):
        if len(self.key_id) == 0:
            return None
        key = self.key_id.pop()  # pop the last element to make the model train on the latest data
        item = self.buffer[self.key].get(key, None)
        if item is None:
            return
        data_size = item[self.size_idx]
        return key, data_size

    def get_data(self, key: str, data_ids: List[str]):
        data = self.buffer[key]
        return [data.get(id_, None) for id_ in data_ids]


def get_ip():
    if "SLURM_JOB_ID" not in os.environ:
        # assume the trainer and the data generator are in the same machine
        return "127.0.0.1"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0)
    try:
        # doesn't even have to be reachable
        s.connect(("10.254.254.254", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

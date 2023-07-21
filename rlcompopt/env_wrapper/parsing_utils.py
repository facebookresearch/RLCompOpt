
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sqlite3

import dgl

DB_CREATION_SCRIPT = """
CREATE TABLE IF NOT EXISTS Vocabs (
    token TEXT NOT NULL UNIQUE, 
    PRIMARY KEY (token)
);
"""


def update_networkx_feature(graph_fn, idx, feature_name, feature):
    graph_fn[idx][feature_name] = feature


def convert_networkx_to_dgl(
    graph, node_attrs=["text_idx", "type"], edge_attrs=["flow", "position"]
):
    return dgl.from_networkx(graph, node_attrs=node_attrs, edge_attrs=edge_attrs)


# utility for feature extraction
class FeatureExtractor:
    def __init__(self, vocab_db_path, online_update=False, graph_version=0):
        self.node_feature_list = ["text", "type", "function", "block"]
        self.node_feature_list_dgl = ["text_idx", "type", "function", "block"]
        self.edge_feature_list = ["flow", "position"]
        self.online_update = online_update
        self.graph_version = graph_version

        if vocab_db_path is None:
            self.vocab_mapping = {}
        else:
            self.connection = sqlite3.connect(vocab_db_path, timeout=3200)
            self.cursor = self.connection.cursor()
            self.cursor.executescript(DB_CREATION_SCRIPT)
            self.connection.commit()
            # Load the dataset
            # FIXME: does "select token from Vocabs;" give the same order each time it is called?
            self.vocabs = list(self.cursor.execute("select token from Vocabs;"))
            v2i = {v[0]: i for i, v in enumerate(self.vocabs)}
            self.vocab_mapping = {"text": v2i}
            if not online_update:
                self.connection.close()

    def save_vocab_to_db(self, cursor, table_name):
        tuple_vers = [v for v in self.vocabs]
        cursor.executemany(f"INSERT OR IGNORE INTO {table_name} VALUES (?)", tuple_vers)

    def process_nx_graph(self, graph):
        """
        Handles all of the requirements of taking a networkx graph and converting it into a
        dgl graph
        Inputs:
            - graph: the networkx graph
            - vocab: the vocabulary, a mapping from word to index.
            - node_feature_list: a list of textual features from the networkx node that we want to make sure
                are featurizable into a vector.
            - edge_feature_list: a list of textual features from the networkx edges that we want to make sure
                are featurizable into a vector.
        """
        self.update_graph_with_vocab(
            graph.nodes, self.node_feature_list, self.vocab_mapping, "nodes"
        )
        # No need to update edge feature: it will not change anything
        # self.update_graph_with_vocab(graph.edges, self.edge_feature_list, self.vocab_mapping, "edges")

        dgl_graph = convert_networkx_to_dgl(
            graph,
            node_attrs=self.node_feature_list_dgl,
            edge_attrs=self.edge_feature_list,
        )
        return dgl_graph

    def update_vocabs(self, token):
        # add a new token into vocab database
        self.cursor.execute("INSERT OR IGNORE INTO Vocabs VALUES (?)", (token,))
        self.connection.commit()
        self.vocabs = list(self.cursor.execute("select token from Vocabs;"))
        v2i = {v[0]: i for i, v in enumerate(self.vocabs)}
        self.vocab_mapping = {"text": v2i}
        assert token in v2i

    def update_graph_with_vocab(self, graph_fn, features, vocab, graph_fn_type="edges"):
        for feature_name in features:
            _counter = 0
            _total = 0
            curr_vocab = None
            if feature_name in vocab:
                curr_vocab = vocab[feature_name]
            len_curr_vocab = len(curr_vocab) if curr_vocab is not None else 0
            for graph_item in graph_fn(data=feature_name):
                feature = graph_item[
                    -1
                ]  # for networkX graph, the node/edge feature is always the last item
                if graph_fn_type == "nodes":
                    idx = graph_item[0]
                else:
                    # for this MultiDiGraph, this is at most one edge for a pair of nodes, so the third idx is 0;
                    # the first two idx are the node idx for this edge
                    idx = graph_item[:-1] + (0,)

                _total += 1
                if feature_name in vocab:
                    # this is for nodes feature "text", convert this feature to idx for embedding later
                    # aggregate all functions to a single type
                    if (
                        self.graph_version == 1
                        and feature.endswith(")")
                        and feature.find(" (") >= 0
                    ):
                        feature = "__function__"
                    token_idx = curr_vocab.get(feature, len_curr_vocab)
                    if (
                        feature_name == "text"
                        and self.online_update
                        and token_idx == len_curr_vocab
                    ):
                        # add this word to vocab database
                        self.update_vocabs(feature)
                        # update curr_vocab
                        curr_vocab = self.vocab_mapping["text"]
                        token_idx = curr_vocab.get(feature, len_curr_vocab)
                    update_networkx_feature(
                        graph_fn, idx, f"{feature_name}_idx", token_idx
                    )
                    if token_idx < len_curr_vocab:
                        _counter += 1
                elif isinstance(feature, str):
                    # this is for nodes feature "text", vocab is empty (it has not been created yet), so save a dummy value
                    assert len(vocab) == 0 and feature_name == "text"
                    update_networkx_feature(graph_fn, idx, f"{feature_name}_idx", -1)
                else:
                    assert isinstance(
                        feature, int
                    ), f"{(feature_name, feature)} is not an int"
            # if feature_name == "text":
            #     print(f"Found {_counter} among {_total} queries, query success rate: {_counter / _total}")

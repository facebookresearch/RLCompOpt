
# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_scatter import scatter_mean, scatter_sum
from torch_geometric.nn.conv import GATv2Conv

from rlcompopt.cl.models.edge_attn import EdgeEncoding, EdgeAttn
from rlcompopt.cl.models.graph_encoders import (  # noqa
    GatedGraphConv,
    GatedGraphConvAttn,
    GraphAttn,
    GraphAttnWin,
    SimpleMessagePassing,
)
from rlcompopt.cl.models.conv import GCNConv, GINConv


MODELS = {
    "GatedGraphConv": GatedGraphConv, 
    "GatedGraphConvAttn": GatedGraphConvAttn, 
    "GraphAttn": GraphAttn, 
    "GraphAttnWin": GraphAttnWin, 
    "EdgeAttn": EdgeAttn,
}
MODELS2 = {
    "GAT": GATv2Conv,
    "GCN": GCNConv,
    "GIN": GINConv,
}


class GNNEncoder(nn.Module):
    def __init__(
        self,
        node_vocab_size,
        node_hidden_size,
        use_node_embedding=True,
        use_edge_embedding=False,
        use_action_embedding=False,
        n_steps=1,
        n_etypes=3,
        n_message_passes=0,
        reward_dim=1,
        use_autophase=True,
        ssl=False,
        autophase_dim=56,
        autophase_bounds=None,
        num_actions=124,
        gnn_type="GatedGraphConv",
        aggr='add',
        heads=None,
        feat_drop=0.0,
        concat_intermediate=False,
        discount_factor=0.99,
        zero_terminal_reward=False,
        update_frequence=100,  # dummy argument, update is controlled in the train script
        mode="pyg",
        graph_version=0,
        edge_emb_dim=0,
        max_edge_position=64,
        node_level_action=False,
        bootstrap_q_learning=False,
        use_subgraph_feature=False,
        subgraph="function",
        divided_by_this_ir=False,  # for eval model
        use_fc=False,
        use_relu=False,
        use_reversed_edge=False,
        use_flow_embedding=False,
        on_policy_gradient=False,
        entropy_factor=0.0003,
        use_history=False,
        use_reward_history=False,
        history_trans_heads=4,
        use_value_function=False,
        use_ppo=False,
        clip_ratio=0.2,
        target_kl=0.01,
        num_local_updates=1,
        use_reinforce=False,
        use_reward_only=False,
        use_reward_and_graph=False,
        use_cl=False,  # contrastive SSL
        ema_momentum=0.99,
        temperature=0.07,
        action_dim=32,
        logit_temperature=1,
        avg_instruct_nodes=False,
        num_heads: int = 1,
        adv_factor: float = 10.,
        no_state_obs=False,
        label_smoothing=0,
        dense_label=False,  # for behavior cloning
        type_graph=False,
        random_mixup=False,
        loss_mixup_coef=0,
        norm_for_cls=False,
        action_histogram_steps=0,  # will be used in data generator
        action_histogram_for_values=False,
        zero_edge_emb=False,
    ):
        super(GNNEncoder, self).__init__()

        self.use_node_embedding = use_node_embedding
        self.use_edge_embedding = use_edge_embedding
        self.max_edge_position = max_edge_position
        self.use_action_embedding = use_action_embedding
        self.node_vocab_size = node_vocab_size
        self.node_hidden_size = node_hidden_size
        self.n_steps = n_steps
        self.n_etypes = n_etypes
        self.n_message_passes = n_message_passes
        self.num_actions = num_actions
        self.reward_dim = reward_dim
        self.gnn_type = gnn_type
        self.heads = heads
        self.feat_drop = feat_drop
        self.concat_intermediate = concat_intermediate
        self.use_autophase = use_autophase
        self.autophase_dim = autophase_dim
        self.autophase_bounds = autophase_bounds
        self.gamma = discount_factor
        self.zero_terminal_reward = zero_terminal_reward
        self.ssl = ssl
        self.graph_version = graph_version
        self.mode = mode
        self.node_level_action = node_level_action
        self.bootstrap_q_learning = bootstrap_q_learning
        self.use_subgraph_feature = use_subgraph_feature
        self.subgraph = subgraph
        self.divided_by_this_ir = divided_by_this_ir
        self.use_fc = use_fc
        self.use_relu = use_relu
        self.use_reversed_edge = use_reversed_edge
        self.use_flow_embedding = use_flow_embedding
        self.on_policy_gradient = on_policy_gradient
        self.entropy_factor = entropy_factor
        self.use_history = use_history
        self.use_reward_history = use_reward_history
        self.history_trans_heads = history_trans_heads
        self.use_value_function = use_value_function
        self.use_ppo = use_ppo
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.num_local_updates = num_local_updates
        self.use_reward_only = use_reward_only
        self.use_reinforce = use_reinforce
        self.use_reward_and_graph = use_reward_and_graph
        self.use_cl = use_cl
        self.ema_momentum = ema_momentum
        self.T = temperature
        self.pred_act_dim = action_dim
        self.online_q_learning = False
        self.logit_temperature = logit_temperature
        self.avg_instruct_nodes = avg_instruct_nodes
        self.adv_factor = adv_factor
        self.no_state_obs = no_state_obs
        self.label_smoothing = label_smoothing
        self.dense_label = dense_label
        self.type_graph = type_graph
        self.random_mixup = random_mixup
        self.loss_mixup_coef = loss_mixup_coef
        self.norm_for_cls = norm_for_cls
        self.use_action_histogram = action_histogram_steps > 0
        self.action_histogram_steps = action_histogram_steps
        self.action_histogram_for_values = action_histogram_for_values
        self.zero_edge_emb = zero_edge_emb
        if self.use_action_histogram:
            assert self.use_history
        if self.action_histogram_for_values:
            assert self.use_action_histogram
            assert self.use_value_function

        if self.ssl:
            assert not use_autophase and not use_action_embedding
            assert use_node_embedding
            self.concat_intermediate = False

        # make sure these attributes exist to feed into 'forward' method
        self.ggcnn = self.gnn = self.Q = self.node_embedding = None
        if self.use_node_embedding:
            num_embedding = node_vocab_size
            self.node_embedding = nn.Embedding(num_embedding, node_hidden_size)

        embed_dim = self.node_hidden_size

        if self.use_autophase and not self.use_reward_only:
            # assert not self.use_history, "Not implemented for autophase with history"
            # hack: autophase to Q input
            self.autophase_nn = nn.Sequential(*[
                nn.Linear(autophase_dim, embed_dim),
                nn.ReLU(),
            ])
            for i in range(self.n_message_passes - 1):
                self.autophase_nn.append(nn.Linear(embed_dim, embed_dim))
                self.autophase_nn.append(nn.ReLU())
            self.gnn_type = None
            if self.use_history:
                assert self.use_action_histogram

        if self.type_graph:
            self.n_type_gnn_layers = 3
            self.type_graph_gnn = nn.ModuleList(
                [
                    SimpleMessagePassing(
                        out_channels=self.node_hidden_size,
                        act=i < self.n_type_gnn_layers - 1,
                    )
                    for i in range(self.n_type_gnn_layers)
                ]
            )

        if self.gnn_type in MODELS:
            GNN = MODELS[gnn_type]
            self.ggcnn = nn.ModuleList(
                [
                    GNN(
                        out_channels=self.node_hidden_size,
                        edge_dim=edge_emb_dim,
                        num_layers=self.n_steps,
                        aggr=aggr,
                        use_reversed_edge=self.use_reversed_edge,
                        num_heads=num_heads,
                        zero_edge_emb=self.zero_edge_emb,
                        **({'shift_win': i % 2} if self.gnn_type == "GraphAttnWin" else {})
                    )
                    for i in range(self.n_message_passes)
                ]
            )
            if gnn_type == "EdgeAttn":
                self.use_edge_embedding = True

        elif self.gnn_type in MODELS2:
            GNNLayer = MODELS2[self.gnn_type]

            if self.use_edge_embedding:
                edge_dim = edge_emb_dim
                fill_value = torch.zeros(edge_dim)
            else:
                edge_dim = None
                fill_value = 'mean'
            self.gnn = nn.ModuleList()

            if self.gnn_type in ["GIN", "GCN"]:
                out_channels=None,
                gnn_kwargs = dict(
                    edge_dim=edge_dim,
                    aggr=aggr
                )
            elif self.gnn_type in ["GAT"]:
                out_channels = self.node_hidden_size // num_heads  # will concat all heads
                gnn_kwargs = dict(
                    out_channels=out_channels,
                    heads=num_heads,
                    concat=True,
                    # negative_slope=0.0,
                    add_self_loops=True,
                    edge_dim=edge_dim,
                    fill_value=fill_value,
                    bias=True,
                )
                self.norms = nn.ModuleList(
                    nn.BatchNorm1d(self.node_hidden_size) for i in range(self.n_message_passes)
                )

            for i in range(self.n_message_passes):
                self.gnn.append(
                    GNNLayer(
                        in_channels=self.node_hidden_size,
                        **gnn_kwargs,
                    )
                )
            embed_dim = self.node_hidden_size

        if self.use_edge_embedding:
            self.edge_encoder = EdgeEncoding(edge_emb_dim)

        if self.use_fc:
            self.fc = nn.ModuleList(
                [
                    torch.nn.Linear(self.node_hidden_size, self.node_hidden_size)
                    for _ in range(self.n_message_passes)
                ]
            )
        else:
            self.fc = None
        if self.use_relu:
            if isinstance(self.use_relu, str):
                self.act_func = eval(self.use_relu)()
            else:
                self.act_func = nn.ReLU()

        if self.ssl:
            self.predictor = nn.Sequential(
                nn.Linear(2 * embed_dim, self.node_hidden_size),
                nn.ReLU(),
                nn.Linear(self.node_hidden_size, 1),
                nn.Sigmoid(),
            )
            self.node_predictor = nn.Sequential(
                nn.Linear(embed_dim, self.node_hidden_size),
                nn.ReLU(),
                nn.Linear(
                    self.node_hidden_size, node_vocab_size - 1
                ),  # there is 1 idx for masked node in node_vocab_size
            )
            return

        if self.use_history:
            if self.use_reward_only or self.use_reward_and_graph:
                lstm_layers = 2
                self.reward_embed = nn.Embedding(4, embed_dim)
                self.mem = nn.LSTM(embed_dim, embed_dim, lstm_layers, batch_first=True)
                self.init_mem_hidden_0 = nn.Parameter(
                    torch.randn(lstm_layers, 1, embed_dim)
                )
                self.init_mem_hidden_1 = nn.Parameter(
                    torch.randn(lstm_layers, 1, embed_dim)
                )
            if self.use_action_histogram:
                self.action_hist_encoder = nn.Sequential(
                    nn.Linear(self.num_actions, self.node_hidden_size),
                    nn.ReLU(),
                    nn.Linear(self.node_hidden_size, self.node_hidden_size),
                )

        if self.use_value_function and not self.use_autophase:
            value_function0 = copy.deepcopy(self.ggcnn or self.gnn)
            self.num_shared_layers = len(value_function0) // 2
            assert self.num_shared_layers < len(value_function0)
            self.value_function0 = value_function0[self.num_shared_layers:]
            value_fc = copy.deepcopy(self.fc)
            self.value_fc = value_fc[self.num_shared_layers:] if self.use_fc else value_fc

            dim_ = embed_dim + edge_emb_dim if gnn_type == "EdgeAttn" else embed_dim
            if self.action_histogram_for_values:
                self.action_hist_encoder_value = copy.deepcopy(self.action_hist_encoder)
                dim_ += self.node_hidden_size

            self.value_function1 = nn.Sequential(
                nn.Linear(dim_, self.node_hidden_size),
                nn.ReLU(),
                nn.Linear(self.node_hidden_size, 1),
            )

        if self.use_value_function and self.use_autophase:
            self.value_function0 = copy.deepcopy(self.autophase_nn)
            dim_ = embed_dim
            if self.action_histogram_for_values:
                self.action_hist_encoder_value = copy.deepcopy(self.action_hist_encoder)
                dim_ += self.node_hidden_size

            self.value_function1 = nn.Linear(dim_, 1)

        if self.use_cl:
            self.projection_head = nn.Sequential(
                nn.Linear(embed_dim, self.node_hidden_size),
                nn.ReLU(),
                nn.Linear(self.node_hidden_size, embed_dim),
            )

            # the prediction head maps (state, action_embedding) to next_state
            self.pred_action_embed = torch.nn.Embedding(
                self.num_actions, self.pred_act_dim
            )
            self.prediction_head = nn.Sequential(
                nn.Linear(embed_dim + self.pred_act_dim, self.node_hidden_size),
                nn.ReLU(),
                nn.Linear(self.node_hidden_size, embed_dim),
            )

        # predict Q function
        multiplier = 2 if self.use_reward_and_graph else 1
        action_histogram_dim = 0
        if self.use_action_histogram:
            action_histogram_dim = self.node_hidden_size
        q_dim = embed_dim * multiplier + action_histogram_dim
        if gnn_type == "EdgeAttn":
            q_dim += edge_emb_dim
        self.Q = nn.Sequential(
            nn.Linear(q_dim, self.node_hidden_size * multiplier),
            nn.ReLU(),
            nn.Linear(self.node_hidden_size * multiplier, self.num_actions),
        )
        if self.bootstrap_q_learning:
            self.Q2 = copy.deepcopy(self.Q)
            self.node_embedding2 = copy.deepcopy(self.node_embedding)
            self.ggcnn2 = copy.deepcopy(self.ggcnn)
            self.gnn2 = copy.deepcopy(self.gnn)
            self.update_target_nets()

        self.mse_loss = nn.MSELoss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        print(f"Embed dim: {embed_dim}")

    def update_target_nets(self):
        if not self.bootstrap_q_learning:
            return
        online_nets = [self.Q, self.node_embedding, self.ggcnn, self.gnn]
        target_nets = [self.Q2, self.node_embedding2, self.ggcnn2, self.gnn2]
        for online, target in zip(online_nets, target_nets):
            if online is not None:
                target.load_state_dict(online.state_dict())
                # freeze params
                target.eval()
                for p in target.parameters():
                    p.requires_grad = False

    def load_state_dict(self, state_dict, strict: bool = True):
        # overwrite super class's method to ensure target nets' parameters are same as online nets
        msg = super().load_state_dict(state_dict, strict)
        self.update_target_nets()
        return msg, {"update_target_nets": "success"}

    def head_params(self):
        return list(self.Q.parameters())

    def freeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.Q.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.parameters():
            p.requires_grad = True

    def reset_parameters(self):
        for module in self.modules():
            if hasattr(module, "weight"):
                module.reset_parameters()

    def forward(
        self,
        g,
        ggcnn=None,
        gnn=None,
        q_net=None,
        autophase=None,
        get_node_src_ids: torch.Tensor = None,
        get_node_dst_ids: torch.Tensor = None,
        get_masked_nodes: torch.Tensor = None,
        freeze_backbone=False,
        action_rewards=None,
        padding_mask=None,
        reward_sign_hist=None,
        seq_pos=None,
    ):
        # feed networks are arguments so that we can pass two different sets of nets (target/online)

        # Set default
        if ggcnn is None and self.ggcnn is not None:
            ggcnn = self.ggcnn
        if gnn is None and self.gnn is not None:
            gnn = self.gnn
        if q_net is None and self.Q is not None:
            q_net = self.Q

        if self.use_reward_only or self.use_reward_and_graph:
            bz, seq_l = reward_sign_hist.shape
            mem_hidden = (
                self.init_mem_hidden_0.expand(-1, bz, -1).contiguous(),
                self.init_mem_hidden_1.expand(-1, bz, -1).contiguous(),
            )  # [n_layer, bz, hidden]
            reward = self.reward_embed(reward_sign_hist)  # [bz, seq_len, hidden]
            out_, _ = self.mem(reward, mem_hidden)  # [bz, seq_len, hidden]
            hdim = out_.shape[-1]
            # seq_pos: [bz]
            seq_pos = (
                seq_pos.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, hdim)
            )  # [bz, 1, hidden]
            reward_feat = torch.gather(out_, dim=1, index=seq_pos).squeeze(
                1
            )  # [bz, 1, hidden] -> [bz, hidden]
            if self.use_reward_only:
                logits = q_net(reward_feat)
                return logits, None

        if self.use_autophase and not self.use_ppo:
            feat_ = self.autophase_nn(autophase)
            if self.no_state_obs:
                feat_ = feat_ * 0
            preds = q_net(feat_)
            return preds, None
        
        if self.use_autophase and self.use_ppo:
            assert self.use_value_function
            feat_ = self.autophase_nn(autophase)
            feat2 = self.value_function0(autophase)
            if self.use_history:
                feat_ = self.encode_action_histogram(
                    feat_, action_rewards, self.action_hist_encoder
                )
            preds = q_net(feat_)
            if self.action_histogram_for_values:
                feat2 = self.encode_action_histogram(
                    feat2, action_rewards, self.action_hist_encoder_value
                )
            values = self.value_function1(feat2)
            return preds, (values, None)

        self.featurize_nodes(g)

        if self.type_graph:
            self.encode_type_graph(g)

        edge_feat = None
        if self.use_edge_embedding:
            edge_feat = self.get_edge_encoding(g)

        edge_index = g.edge_index

        # get control flow edges
        ctrl_flow = None
        if self.use_reversed_edge == 1:
            ctrl_flow = g["flow"].flatten() == 0
        elif self.use_reversed_edge == 2:
            ctrl_flow = g["flow"].flatten() <= 1
        node2func = g['function'].flatten()
        node2block = g['block'].flatten()

        pos = None
        ordered_instr_idx = None
        if self.gnn_type == "GraphAttnWin":
            ordered_instr_idx = g['ordered_instr_idx']
            graph_idx = g.batch[ordered_instr_idx].unsqueeze(-1)
            func_idx = g['function'][ordered_instr_idx]
            blk_idx = g['block'][ordered_instr_idx]
            blk_pos = g['blk_pos'].unsqueeze(-1)
            pos = torch.cat([graph_idx, func_idx, blk_idx, blk_pos], dim=1)

        edge_attr = edge_feat
        if self.gnn_type in MODELS2:
            if self.use_reversed_edge:
                back_edge_index = edge_index.T[ctrl_flow].T
                edge_index = torch.cat([edge_index, back_edge_index], dim=1)
            # edge_attr = self.get_edge_embedding(g, edge_index, g.edge_index.shape[1])

        res = g["feat"]

        layers = ggcnn or gnn
        res, edge_feat, res_mid, edge_feat_mid = self.layers_encode(
            res, edge_feat, layers, self.fc, edge_index, g["flow"], ctrl_flow,
            node2func, node2block, pos, ordered_instr_idx, edge_attr,
        )

        g["feat"] = res

        if self.use_value_function:
            res_, edge_feat_, _, _ = self.layers_encode(
                res_mid, edge_feat_mid, self.value_function0, self.value_fc, edge_index, g["flow"], ctrl_flow,
                node2func, node2block, pos, ordered_instr_idx, edge_attr,
            )
            graph_agg_v = scatter_mean(res_, g.batch, dim=0, dim_size=g.num_graphs)
            if self.gnn_type == "EdgeAttn":
                edges_batch_idx = g.batch[g.edge_index[0]]
                edge_agg_v = scatter_mean(edge_feat_, edges_batch_idx, dim=0, dim_size=g.num_graphs)
                graph_agg_v = torch.cat([graph_agg_v, edge_agg_v], dim=1)
            if self.action_histogram_for_values:
                graph_agg_v = self.encode_action_histogram(
                    graph_agg_v, action_rewards, self.action_hist_encoder_value
                )
            values = self.value_function1(graph_agg_v).flatten()  # [bz]

        graph_repr = None
        if not self.ssl:
            if self.concat_intermediate and self.gnn_type in MODELS:
                pass  # 'concat_intermediate' is deplicated
            else:
                if self.use_subgraph_feature:
                    if "subgraph_node_idx" not in g.keys and (
                        not hasattr(g, "batch") or g.batch.sum().item() == 0
                    ):  # ensure no batch or batch contains only 1 graph
                        g["subgraph_node_idx"] = g[self.subgraph]
                        num_subgraphs = g["subgraph_node_idx"].max().item() + 1
                        g["subgraph_idx"] = torch.zeros(
                            num_subgraphs, dtype=torch.long, device=g["feat"].device
                        )
                        feat, count = torch.unique(
                            g[self.subgraph].flatten(), sorted=True, return_counts=True
                        )
                        g["subgraph_num_nodes"] = count / g.num_nodes
                    graph_agg = scatter_mean(g["feat"], g["subgraph_node_idx"], dim=0, dim_size=g.num_graphs)
                else:
                    if self.avg_instruct_nodes:
                        instruct_node = g["type"].flatten() == 0
                        assert g.batch.ndim == 1
                        batch = g.batch[instruct_node]
                        feat = g["feat"][instruct_node]
                        graph_agg = scatter_mean(feat, batch, dim=0, dim_size=g.num_graphs)
                        if self.random_mixup and self.training:
                            normalizer = scatter_sum(torch.ones_like(batch), batch, dim=0, dim_size=g.num_graphs)
                    else:
                        graph_agg = scatter_mean(g["feat"], g.batch, dim=0, dim_size=g.num_graphs)
                        if self.random_mixup and self.training:
                            normalizer = scatter_sum(torch.ones_like(g.batch), g.batch, dim=0, dim_size=g.num_graphs)
                    if self.use_cl:
                        graph_repr = self.projection_head(graph_agg)

                if self.random_mixup and self.training:
                    bz = graph_agg.shape[0]
                    num_mix = int(self.random_mixup * bz)
                    num_perm = (num_mix // bz) + 1
                    mix_idx0 = [torch.randperm(bz, device=graph_agg.device) for _ in range(num_perm)]
                    mix_idx = torch.cat(mix_idx0)[:num_mix]
                    self.mix_idx = mix_idx  # save for label mix
                    self.num_mix = num_mix
                    self.num_perm =num_perm

                    norm1 = torch.cat([normalizer] * num_perm)[:num_mix, None]
                    norm2 = normalizer[mix_idx][:, None]

                    graph1 = torch.cat([graph_agg] * num_perm)[:num_mix]
                    graph2 = graph_agg[mix_idx]

                    mixed = (graph1 * norm1 + graph2 * norm2) / (norm1 + norm2)
                    graph_agg = torch.cat([graph_agg, mixed], dim=0)

                if self.gnn_type == "EdgeAttn":
                    edges_batch_idx = g.batch[g.edge_index[0]]
                    edge_agg = scatter_mean(edge_feat, edges_batch_idx, dim=0, dim_size=g.num_graphs)

                    if self.random_mixup and self.training:
                        normalizer_e = scatter_sum(torch.ones_like(edges_batch_idx), edges_batch_idx, dim=0, dim_size=g.num_graphs)
                        norm1 = torch.cat([normalizer_e] * num_perm)[:num_mix, None]
                        norm2 = normalizer_e[mix_idx][:, None]

                        e1 = torch.cat([edge_agg] * num_perm)[:num_mix]
                        e2 = edge_agg[mix_idx]

                        mixed_e = (e1 * norm1 + e2 * norm2) / (norm1 + norm2)
                        edge_agg = torch.cat([edge_agg, mixed_e], dim=0)
                    graph_agg = torch.cat([graph_agg, edge_agg], dim=1)

            if self.no_state_obs:
                g["feat"] = g["feat"] * 0
                graph_agg = graph_agg * 0

            if self.node_level_action:
                res = q_net(g["feat"])
                res = scatter_sum(res, g.batch, dim=0, dim_size=g.num_graphs)
            else:
                if self.use_history:
                    assert not self.random_mixup
                    if self.use_reward_and_graph:
                        graph_agg = torch.cat([graph_agg, reward_feat], dim=-1)
                    if self.use_action_histogram:
                        graph_agg = self.encode_action_histogram(
                            graph_agg, action_rewards, self.action_hist_encoder
                        )

                res = q_net(graph_agg)
                if self.use_subgraph_feature:
                    assert not self.use_reward_and_graph, "Not implemented"
                    res = res * g["subgraph_num_nodes"].view(
                        -1, 1
                    )  # subgraph_num_nodes is already normalized
                    res = scatter_sum(res, g["subgraph_idx"], dim=0)
            if self.use_value_function:
                return res, (values, graph_repr)
            return res, (graph_agg, graph_repr)
        else:
            x = g["feat"]
            if get_masked_nodes is not None:
                get_masked_nodes = x[get_masked_nodes]
            return x[get_node_src_ids], x[get_node_dst_ids], get_masked_nodes

    def layers_encode(
        self, 
        res,
        edge_feat,
        layers, 
        fc,
        edge_index, 
        g_flow,
        ctrl_flow,
        node2func,
        node2block,
        pos,
        ordered_instr_idx,
        edge_attr,
    ):
        res_mid, edge_feat_mid = None, None
        n_layers = len(layers)
        for i, layer in enumerate(layers):
            if self.gnn_type in MODELS:
                res = layer(
                    res,
                    edge_index,
                    edge_attr=edge_feat,
                    flow=g_flow,
                    key_edge_mask=ctrl_flow,
                    node2func=node2func, 
                    node2block=node2block,
                    pos=pos,
                    ordered_instr_idx=ordered_instr_idx,
                )
            elif self.gnn_type in MODELS2:
                res = layer(res, edge_index, edge_attr=edge_attr).flatten(1)  # for GAT, need to flatten
            if self.gnn_type == "EdgeAttn":
                res, edge_feat = res

            if self.gnn_type in ["GAT"]:
                res = self.norms[i](res)

            if self.use_relu and i - 1 < n_layers:
                res = self.act_func(res)
            if self.use_fc:
                res = fc[i](res)
                if self.use_relu and i - 1 < n_layers:
                    res = self.act_func(res)

            if self.feat_drop:
                res = F.dropout(res, p=self.feat_drop, training=self.training)
                if edge_feat is not None:
                    edge_feat = F.dropout(edge_feat, p=self.feat_drop, training=self.training)

            if self.use_value_function and i == self.num_shared_layers:
                res_mid, edge_feat_mid = res, edge_feat
        return res, edge_feat, res_mid, edge_feat_mid

    def encode_action_histogram(self, state_feat, action_rewards, action_hist_encoder):
        """Encode action histogram and concatenate it to state_feat """
        actions_histogram, rewards = action_rewards  # [batchsize, num_actions], None
        actions_feat = action_hist_encoder(actions_histogram)
        out = torch.cat([state_feat, actions_feat], dim=1)
        return out

    def get_action(self, g):
        preds, _ = self.forward(g)
        maxQ, action = preds.max(dim=1)
        return action

    def featurize_nodes(self, g):
        # This is very CompilerGym specific, can be rewritten for other tasks
        features = []
        if self.use_node_embedding:
            features.append(self.node_embedding(g["x"].flatten()))

        g["feat"] = torch.cat(features)

    def get_edge_embedding(self, g, edge_index, num_origin_edges=None):
        if self.use_edge_embedding:
            function = g["function"].flatten()
            block = g["block"].flatten()
            i = edge_index[0]
            j = edge_index[1]
            func_diff = function[i] - function[j]
            func_diff = func_diff.bool().float().unsqueeze(-1)
            blk_diff = block[i] - block[j]
            blk_diff = blk_diff.bool().float().unsqueeze(-1)
            edge_attr = torch.cat([func_diff, blk_diff], dim=1)
            if self.use_reversed_edge:
                back = torch.ones_like(blk_diff)
                back[:num_origin_edges] = 0.
                edge_attr = torch.cat([edge_attr, back], dim=1)
            return edge_attr

    def cl_loss(self, graph_repr_, actions, buffer, curr_repr_idx, next_repr_idx):
        """
        A forward dynamic model
        Input:
            graph_repr_: the representation of the batched states from the latest encoder [batchsize, dim]
            actions: the action taken [batchsize]
            buffer: a tensor of shape [num_samples, repr_dim]
            curr_repr_idx: the indices of the current state representation (from the EMA encoder) [batchsize]
            next_repr_idx: the indices of the next state representation (from the EMA encoder) [batchsize]
        Output:
            cross entropy loss of finding the next state using the current state
        """
        action_embed = self.pred_action_embed(actions)
        feat = torch.cat([graph_repr_, action_embed], dim=-1)
        curr_repr = self.prediction_head(feat)
        curr_repr = F.normalize(curr_repr, p=2, dim=-1)

        # buffer is already normalized
        paired_similarity = curr_repr @ buffer.T  # [batchsize, num_samples]

        # apply temperature
        logits0 = paired_similarity / self.T

        logits0 = logits0.clone()
        logits = torch.scatter(
            logits0, dim=1, index=curr_repr_idx.view(-1, 1), value=-100.0
        )

        # cross entropy loss
        loss = self.ce_loss(logits, next_repr_idx)
        return loss

    def encode_type_graph(self, g):
        """
        Encode type graphs (subgraph with type nodes). And then remove the edges in the subgraphs.
        Should be after node embedding but before other operations
        """
        edge_index = g.edge_index
        node_type = g["type"].flatten()
        src_type = node_type[edge_index[0]]
        tgt_type = node_type[edge_index[1]]
        mask = (src_type == 3) | (tgt_type == 3)
        edge_index = edge_index.T
        type_graph_edges = edge_index[mask].T
        res = g["feat"].clone()
        for i, layer in enumerate(self.type_graph_gnn):
            res = layer(res, type_graph_edges)
        update_mask = (node_type == 1) | (node_type == 2)  # only update var and val
        feat = g["feat"].clone()
        feat[update_mask] = res[update_mask]
        g["feat"] = feat

        non_type_mask = ~(mask | (g["flow"].flatten() == 2))  # remove call edges
        g.edge_index = edge_index[non_type_mask].T
        g["flow"] = g["flow"][non_type_mask]
        g["position"] = g["position"][non_type_mask]

        # get control subgraphs
        # subset = node_type != 3
        # edge_attr = torch.cat([g["flow"], g["position"]], dim=1)
        # new_edge_index, edge_attr = subgraph(subset, g.edge_index, edge_attr, relabel_nodes=True)
        # g["flow"], g["position"] = torch.split(edge_attr, 1, dim=1)
        # for attr in ["type", "block", "function"]:
        #     g[attr] = g[attr][subset]
        # g.batch = g.batch[subset]
        # g.edge_index = new_edge_index
        # g['x'] = g['x'][subset]
        # g['feat'] = g['feat'][subset]

    def get_edge_encoding(self, g):
        edge_types = g["flow"].flatten()
        edge_pos = g["position"].flatten()
        edge_index = g.edge_index
        block = g["block"].flatten()
        src_block = block[edge_index[0]]
        tgt_block = block[edge_index[1]]
        block_idx = torch.stack([src_block, tgt_block])
        edge_enc = self.edge_encoder(edge_types, edge_pos, block_idx)
        return edge_enc


class QLearner(GNNEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.online_q_learning = True

    def forward(
        self,
        g,
        next_g,
        actions,
        labels,
        raw_rewards,
        non_terminal,
        autophase,
        eps=0.0,
        freeze_backbone=False,
    ):

        preds, _ = super().forward(
            g, autophase=autophase, freeze_backbone=freeze_backbone
        )

        # Picking actions.
        Q_values = preds.gather(1, actions.unsqueeze(1)).squeeze(1)
        # scaled_labels = rescale(labels, eps=eps)
        # inv_scale_pred = inv_rescale(Q_values, eps=eps)

        # print ("preds are: ", inv_scale_pred.shape, labels.shape)
        # print ("diff is: ", ((inv_scale_pred - labels)**2).mean())

        labels2 = labels.clone()
        labels2[labels == 0.0] = 1e-5
        return (
            # self.mse_loss(Q_values, scaled_labels),
            self.mse_loss(Q_values, labels),
            labels.detach(),
            (Q_values.detach() - labels).abs().mean(),
            ((labels2 - Q_values.detach()).abs() / labels2).mean(),
            Q_values.detach(),
        )

    def get_logits(self, g, autophase):
        logits, _ = super().forward(g, autophase=autophase)
        logits = logits / self.logit_temperature
        return logits, None


class CLSLearner(GNNEncoder):
    """
    Perform classifications on actions / action-sequences.
    Including single-class classification, normalized value prediction,
    and raw value regression.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing, reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(
        self,
        g,
        next_g,
        actions,
        labels,
        raw_rewards,
        non_terminal,
        autophase,
        eps=0.0,
        freeze_backbone=False,
    ):
        preds, _ = super().forward(
            g, autophase=autophase, freeze_backbone=freeze_backbone
        )
        if self.random_mixup and self.training:
            assert self.dense_label
            labels = self.mix_label(g, labels)
        if self.dense_label:
            if self.norm_for_cls:
                if not self.norm_for_cls == "mse":
                    labels = self.norm_logits(labels)
                    preds = self.norm_logits(preds)
                loss = self.mse_loss(preds, labels).mean(-1)
            else:
                labels = (labels / self.T).softmax(-1)
                loss = self.criterion(preds, labels)

            if self.random_mixup and self.training:
                loss = loss.clone()
                loss[-self.num_mix:].mul_(self.loss_mixup_coef)
        else:
            loss = self.criterion(preds, actions)
        loss = loss.mean()
        _, y_pred = torch.max(preds, dim=1)
        if not self.dense_label:
            labels = torch.zeros_like(preds)
            labels = torch.scatter(labels, dim=1, index=actions[:, None], value=1)
        inputs = F.log_softmax(preds.clone().detach(), dim=1)
        if self.norm_for_cls == "mse" or torch.any(labels <= 0.):
            kl = None
        else:
            kl = F.kl_div(inputs, labels, reduction="batchmean").item()
        if self.random_mixup and self.training:
            preds = preds[:-self.num_mix]
            y_pred = y_pred[:-self.num_mix]
        return (
            loss,
            kl,
            y_pred.clone().detach(),
            preds.clone().detach(),
        )

    def mix_label(self, g, labels):
        num_perm = self.num_perm
        num_mix = self.num_mix
        mix_idx = self.mix_idx
        norm1 = torch.cat([g["ir_oz"]] * num_perm)[:num_mix, None]
        norm2 = g["ir_oz"][mix_idx][:, None]

        labels1 = torch.cat([labels] * num_perm, dim=0)[:num_mix]
        labels2 = labels[mix_idx]

        mixed = (labels1 * norm1 + labels2 * norm2) / (norm1 + norm2)
        labels_ = torch.cat([labels, mixed], dim=0)
        return labels_

    def norm_logits(self, logits: torch.Tensor):
        assert logits.ndim == 2
        logits = logits - logits.mean(dim=1, keepdim=True)
        logits = F.normalize(logits, dim=1)
        return logits

    def get_logits(self, g, autophase):
        logits, _ = super().forward(g, autophase=autophase)
        if self.dense_label:
            if self.norm_for_cls:
                if not self.norm_for_cls == "mse":
                    logits = self.norm_logits(logits)
            else:
                logits = logits / self.T
        return logits, None


class TDLearner(GNNEncoder):
    def forward(
        self,
        g,
        next_g,
        actions,
        labels,
        raw_rewards,
        non_terminal,
        autophase,
        eps=0.0,
        freeze_backbone=False,
    ):
        """
        Loss function, use simple TD learning. Must set bootstrap_q_learning=True
        """
        preds, _ = super().forward(
            g, self.ggcnn, self.gnn, self.Q, autophase, freeze_backbone=freeze_backbone
        )
        with torch.no_grad():
            preds2, _ = self.forward(
                next_g,
                self.ggcnn2,
                self.gnn2,
                self.Q2,
                autophase,
                freeze_backbone=False,
            )

        next_state_values = preds2.max(dim=1)[0]
        # if next state is a terminal state, set future reward to 0
        if self.zero_terminal_reward:
            tmp = self.gamma * next_state_values * non_terminal
            expected_state_action_values = raw_rewards + tmp
        else:
            expected_state_action_values = raw_rewards + self.gamma * next_state_values

        # Picking actions.
        Q_values = preds.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = self.mse_loss(Q_values, expected_state_action_values.detach().flatten())

        # print ("preds are: ", inv_scale_pred.shape, labels.shape)
        # print ("diff is: ", ((inv_scale_pred - labels)**2).mean())

        expected_state_action_values = expected_state_action_values.clone()
        expected_state_action_values[expected_state_action_values == 0.0] = 1.0
        return (
            # self.mse_loss(Q_values, scaled_labels),
            loss,
            expected_state_action_values,
            (expected_state_action_values - Q_values).abs().mean(),
            (
                (expected_state_action_values - Q_values).abs()
                / expected_state_action_values
            ).mean(),
            Q_values,
        )


class Reinforce(GNNEncoder):
    """
    A simply reinforce algorithm with log barrier regularization
    """

    def forward(
        self,
        g,
        autophase,
        actions,
        labels,
        action_rewards=None,
        padding_mask=None,
        reward_sign_hist=None,
        seq_pos=None,
        buffer=None,
        curr_repr_idx=None,
        next_repr_idx=None,
    ):

        logits, other = super().forward(
            g,
            autophase=autophase,
            action_rewards=action_rewards,
            padding_mask=padding_mask,
            reward_sign_hist=reward_sign_hist,
            seq_pos=seq_pos,
        )

        all_log_probs = F.log_softmax(logits, dim=-1)

        m = Categorical(logits=logits)
        log_probs = m.log_prob(actions)
        with torch.no_grad():
            entropies = m.entropy()
            normalize_entropy = entropies.mean() / np.log(logits.shape[-1])

        policy_loss = (-log_probs * labels).mean()

        log_barrier = all_log_probs.mean() + np.log(logits.shape[-1])
        loss_entropy = -self.entropy_factor * log_barrier

        if self.use_cl and buffer is not None:
            _, graph_repr = other
            cl_loss = self.cl_loss(
                graph_repr, actions, buffer, curr_repr_idx, next_repr_idx
            )
            cl_loss_ = cl_loss.detach().item()
        else:
            cl_loss = 0
            cl_loss_ = 0

        loss = policy_loss + loss_entropy + cl_loss

        return (
            loss,
            policy_loss.detach(),
            loss_entropy.detach(),
            normalize_entropy.detach(),
            cl_loss_,
        )

    def get_logits(
        self,
        g,
        autophase,
        action_rewards=None,
        padding_mask=None,
        reward_sign_hist=None,
        seq_pos=None,
    ):
        logits, other = super().forward(
            g,
            autophase=autophase,
            action_rewards=action_rewards,
            padding_mask=padding_mask,
            reward_sign_hist=reward_sign_hist,
            seq_pos=seq_pos,
        )
        repr_ = None
        if other is not None and other[1] is not None:
            repr_ = F.normalize(other[1], p=2, dim=-1)
        return logits, None, repr_


class PPO(GNNEncoder):
    def pi(self, logits, actions):
        m = Categorical(logits=logits)
        logps = m.log_prob(actions)
        return m, logps

    def compute_loss_pi(self, logits, actions, adv, logp_old=None):
        # Policy loss
        # actions, adv, logp_old should have shape [bz]
        # adv = (adv - adv.mean()) / adv.std()
        adv = adv * self.adv_factor
        pi, logp = self.pi(logits, actions)
        if logp_old is None:  # the first local update is always performed
            logp_old = logp.detach()
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item() / np.log(logits.shape[-1])
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac, logp_old=logp_old)

        return loss_pi, pi_info

    def compute_loss_v(self, values, returns):
        # for computing value loss
        values = values.flatten()
        return ((values - returns) ** 2).mean()

    def forward(
        self, g, autophase, actions, adv, returns, logp_old,
        action_rewards=None,
        padding_mask=None,
        reward_sign_hist=None,
        seq_pos=None,
    ):
        logits, (values, _) = super().forward(
            g,
            autophase=autophase,
            action_rewards=action_rewards,
            padding_mask=padding_mask,
            reward_sign_hist=reward_sign_hist,
            seq_pos=seq_pos,
        )
        loss_pi, pi_info = self.compute_loss_pi(logits, actions, adv, logp_old)
        loss_v = self.compute_loss_v(values, returns)
        kl = pi_info["kl"]
        pi_info["update_pi"] = False if kl > 1.5 * self.target_kl else True

        all_log_probs = F.log_softmax(logits, dim=-1)
        log_barrier = all_log_probs.mean() + np.log(logits.shape[-1])
        loss_entropy = -self.entropy_factor * log_barrier
        pi_info["log_barrier"] = -log_barrier.detach().item()
        return loss_pi, loss_v, pi_info, loss_entropy

    def get_logits(
        self, 
        g, 
        autophase=None,
        action_rewards=None,
        padding_mask=None,
        reward_sign_hist=None,
        seq_pos=None,
    ):
        logits, (values, repr_) = super().forward(
            g, 
            autophase=autophase,
            action_rewards=action_rewards,
            padding_mask=padding_mask,
            reward_sign_hist=reward_sign_hist,
            seq_pos=seq_pos,
        )
        if repr_ is not None:
            repr_ = F.normalize(repr_, p=2, dim=-1)
        return logits, values, repr_

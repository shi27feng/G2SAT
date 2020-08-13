from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch_geometric as pyg


# from torch_geometric.nn import GCNConv

# ###################### NNs #############################


class GCN(torch.nn.Module):
    def _forward_unimplemented(self, *input: Any) -> None:
        pass

    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = pyg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = pyg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([pyg.nn.GCNConv(hidden_dim, hidden_dim)
                                          for i in range(layer_num - 2)])
        self.conv_out = pyg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = fn.relu(x)
        if self.dropout:
            x = fn.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = fn.relu(x)
            if self.dropout:
                x = fn.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = fn.normalize(x, p=2, dim=-1)
        return x

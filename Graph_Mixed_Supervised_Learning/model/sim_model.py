import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch_geometric.nn import GCNConv
from model.moe import MoE
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from torch import Tensor

def pair_enumeration(x):
    '''
        input:  [B,D]
        return: [B*B,D]

        input  [[a],
                [b]]
        return [[a,a],
                [b,a],
                [a,b],
                [b,b]]
    '''
    assert x.ndimension() == 2, 'Input dimension must be 2'
    # [a,b,c,a,b,c,a,b,c]
    # [a,a,a,b,b,b,c,c,c]
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    return torch.cat((x1, x2), dim=1)


def sim_label(data, nodes):
    y_sim = []
    for i in range(60):
        for j in range(60):
            if data.y[nodes[i]] == data.y[nodes[j]]:
                y_sim.append(1)
            else:
                y_sim.append(0)
    return y_sim


class GCNSim(nn.Module):
    def __init__(self, in_feats):
        super(GCNSim, self).__init__()
        self.conv1 = GCNConv(in_feats, 32)
        self.conv2 = GCNConv(32, 16)
        self.relu = nn.ReLU(inplace=True)

        self.fc = MoE(32, 16, num_experts=4, k=1)
        self.fc2 = MoE(16, 2, num_experts=4, k=1)
        # self.fc = nn.Linear(32, 16)
        # self.fc2 = nn.Linear(16, 2)

    def forward(self, data, nodes):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        x_sim = None
        if self.training is True:
            # for i in range(len(nodes):
            for i in range(60):
                for j in range(60):
                    ij_sim = torch.cat((x[nodes[i]], x[nodes[j]]), -1).unsqueeze(0)
                    if x_sim is None:
                        x_sim = ij_sim
                    else:
                        x_sim = torch.cat((x_sim, ij_sim), 0)
        else:
            for i in range(len(nodes)):
                # for j in range(len(nodes)):
                for j in range(50):
                    # if data.y[i] == data.y[j]:
                    ij_sim = torch.cat((x[nodes[i]], x[nodes[j]]), -1).unsqueeze(0)
                    if x_sim is None:
                        x_sim = ij_sim
                    else:
                        x_sim = torch.cat((x_sim, ij_sim), 0)
            # for i in range(len(nodes)):
            #     for j in range(len(nodes)):
            #         x_sim = torch.cat((x_sim, torch.cat((x[i], x[j]), 1)), 0)

        x = x_sim
        x = self.fc(x)
        x = self.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1), x_sim

    def get_l_ratio(self):
        return self.fc.get_l_ratio()*self.fc2.get_l_ratio()

class GCN(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


class GCNLipMoE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCNLipMoE, self).__init__()
        self.conv1 = GCNConv_moe(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, out_feats)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)



class GCNConv_moe(GCNConv):
    def __init__(self, in_channels: int, out_channels: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, num_experts=4, noisy_gating=True, k=1, **kwargs):
        super(GCNConv_moe, self).__init__(in_channels=in_channels, out_channels=out_channels, improved=improved, cached=cached, add_self_loops=add_self_loops, normalize=normalize, bias=bias, **kwargs)
        self.lin = MoE(input_size=in_channels, output_size=out_channels, num_experts=num_experts, noisy_gating=noisy_gating, k=k) # bias=False

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

        x = self.lin(x)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if self.bias is not None:
            x += self.bias

        return x

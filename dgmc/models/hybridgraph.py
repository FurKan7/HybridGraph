import torch
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn import SplineConv,SAGEConv


class HybridGraphModel(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True, lin=True, dropout=0.0):
        super(HybridGraphModel, self).__init__()

        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = nn.ModuleList()

        for i in range(num_layers):
            if i % 2 == 0:
                # Use SplineConv on even layers
                self.convs.append(SplineConv(in_channels, out_channels, dim, kernel_size=5))
            else:
                # Use SAGEConv on odd layers
                self.convs.append(SAGEConv(in_channels, out_channels))
            in_channels = out_channels

        if self.cat:
            in_channels = self.in_channels + num_layers * out_channels
        else:
            in_channels = out_channels

        if self.lin:
            self.out_channels = out_channels
            self.final = Lin(in_channels, out_channels)
        else:
            self.out_channels = in_channels

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.lin:
            self.final.reset_parameters()

    def forward(self, x, edge_index, edge_attr, *args):
        xs = [x]

        for conv in self.convs:
            x = conv(xs[-1], edge_index, edge_attr) if isinstance(conv, SplineConv) else conv(xs[-1], edge_index)
            x = F.relu(x)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, dropout={})').format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.dim, self.num_layers, self.cat, self.lin, self.dropout)
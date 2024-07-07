import os.path as osp
import random

import argparse
import torch
from torch_geometric.data import Data, DataLoader
import torch_geometric.transforms as T
from torch_geometric.datasets import PascalPF
from torch_geometric.nn import SplineConv, GINConv, GCNConv, GATConv, SAGEConv, EdgeConv,GATConv, BatchNorm
from dgmc.models import DGMC, SplineCNN
import torch.nn as nn
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn.pool import EdgePooling as edge_pool
from dgmc.models import MLP  # Assuming MLP is from your earlier files

from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=64)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=200)
args = parser.parse_args()


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
            x = F.elu(x)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, dropout={})').format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.dim, self.num_layers, self.cat, self.lin, self.dropout)

class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True,lin=True, dropout=0.0):
        super(GraphSAGE, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.dropout = dropout
        self.lin = lin

        self.convs = nn.ModuleList()

        # Add GraphSAGE layers
        for i in range(num_layers):
            if i == 0:
                self.convs.append(SAGEConv(in_channels, out_channels))
            else:
                self.convs.append(SAGEConv(out_channels, out_channels))

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
        # print(edge_attr)
        for conv in self.convs:
            xs += [F.relu(conv(xs[-1], edge_index))]

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, cat={}, dropout={})').format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.cat, self.dropout)

class EdgeGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim,  num_layers, cat=True, lin=True, dropout=0.0):
        super(EdgeGNN, self).__init__()

        self.in_channels = in_channels
        self.num_layers = num_layers
        self.cat = cat
        self.dim = dim
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            nn = Lin(2 * in_channels, out_channels)
            conv = EdgeConv(nn, aggr='mean')
            self.convs.append(conv)
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

    def forward(self, x, edge_index, *args):
        xs = [x]

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x

    def __repr__(self):
        return ('{}({}, {}, num_layers={}, cat={}, lin={}, dropout={})').format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_layers, self.cat, self.lin, self.dropout)

class HybridGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dim, num_layers, cat=True, lin=True, dropout=0.0):
        super(HybridGNN, self).__init__()
        

        self.in_channels = in_channels
        self.dim = dim
        self.num_layers = num_layers
        self.cat = cat
        self.lin = lin
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        # Add alternating types of GNN layers
        for i in range(num_layers):
            if i % 2 == 0:  # Even layers (including first layer): SplineConv
                self.convs.append(SplineConv(in_channels, out_channels, dim, kernel_size=5))
            else:  # Odd layers: GINConv
                mlp = MLP(out_channels, out_channels, 4, batch_norm=False, dropout=dropout)
                self.convs.append(GINConv(mlp))
            in_channels = out_channels  # Update in_channels for the next layer

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


class RandomGraphDataset(torch.utils.data.Dataset):
    def __init__(self, min_inliers, max_inliers, min_outliers, max_outliers,
                 min_scale=0.9, max_scale=1.2, noise=0.05, transform=None):

        self.min_inliers = min_inliers
        self.max_inliers = max_inliers
        self.min_outliers = min_outliers
        self.max_outliers = max_outliers
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.noise = noise
        self.transform = transform

    def __len__(self):
        return 1024

    def __getitem__(self, idx):
        num_inliers = random.randint(self.min_inliers, self.max_inliers)
        num_outliers = random.randint(self.min_outliers, self.max_outliers)

        pos_s = 2 * torch.rand((num_inliers, 2)) - 1
        pos_t = pos_s + self.noise * torch.randn_like(pos_s)

        y_s = torch.arange(pos_s.size(0))
        y_t = torch.arange(pos_t.size(0))

        pos_s = torch.cat([pos_s, 3 - torch.rand((num_outliers, 2))], dim=0)
        pos_t = torch.cat([pos_t, 3 - torch.rand((num_outliers, 2))], dim=0)

        data_s = Data(pos=pos_s, y_index=y_s)
        data_t = Data(pos=pos_t, y=y_t)

        if self.transform is not None:
            data_s = self.transform(data_s)
            data_t = self.transform(data_t)

        data = Data(num_nodes=pos_s.size(0))
        # for key in data_s.keys:
        #     data['{}_s'.format(key)] = data_s[key]
        # for key in data_t.keys:
        #     data['{}_t'.format(key)] = data_t[key]
        for key in data_s.keys():  # Corrected to call the method
            data['{}_s'.format(key)] = data_s[key]
        for key in data_t.keys():  # Corrected to call the method
            data['{}_t'.format(key)] = data_t[key]

        return data


transform = T.Compose([
    T.Constant(),
    T.KNNGraph(k=8),
    T.Cartesian(),
])
train_dataset = RandomGraphDataset(30, 60, 0, 20, transform=transform)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

path = osp.join('..', 'data', 'PascalPF')
test_datasets = [PascalPF(path, cat, transform) for cat in PascalPF.categories]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# psi_1 = SplineCNN(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
# psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True,
#                   dropout=0.0)

psi_1 = HybridGraphModel(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
print('psi_1: ',psi_1)
psi_2 = HybridGraphModel(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True, dropout=0.0)
print('psi_2: ',psi_2)                  


# psi_1 = HybridGNN(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
# print('psi_1: ',psi_1)
# psi_2 = HybridGNN(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)
# psi_1 = GraphSAGE(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
# print('psi_1: ',psi_1)
# psi_2 = GraphSAGE(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)

# psi_1 = EdgeGNN(1, args.dim, 2, args.num_layers, cat=False, dropout=0.0)
# print('psi_1: ',psi_1)
# psi_2 = EdgeGNN(args.rnd_dim, args.rnd_dim, 2, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def train():
    model.train()

    total_loss = total_examples = total_correct = 0
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = torch.stack([data.y_index_s, data.y_t], dim=0)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += model.acc(S_L, y, reduction='sum')
        total_examples += y.size(1)

    return total_loss / len(train_loader), total_correct / total_examples


@torch.no_grad()
def test(dataset):
    model.eval()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    correct = num_examples = 0
    times = []
    for pair in dataset.pairs:
        data_s, data_t = dataset[pair[0]], dataset[pair[1]]
        data_s, data_t = data_s.to(device), data_t.to(device)
        start.record()
        S_0, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr, None,
                         data_t.x, data_t.edge_index, data_t.edge_attr, None)

        end.record()
        torch.cuda.synchronize()                 
        y = torch.arange(data_s.num_nodes, device=device)
        y = torch.stack([y, y], dim=0)
        correct += model.acc(S_L, y, reduction='sum')
        num_examples += y.size(1)
        times.append(start.elapsed_time(end))

    # return correct / num_examples
    return sum(times)/len(times), correct / num_examples


for epoch in range(1, 33):
    start = timer()
    loss, acc = train()
    end = timer()


    time = end-start 
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Acc: {acc:.2f},  Time: {time:.2f}')


    accs = [] 
    times = []

    for test_dataset in test_datasets: 
      t, a = test(test_dataset)
      accs.append(100*a)
      times.append(t)

    accs += [sum(accs) / len(accs)]
    times = sum(times)/len(times)
    
    print(' '.join([c[:5].ljust(5) for c in PascalPF.categories] + ['mean']))
    print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
    print('average inference time: ' + str(times))
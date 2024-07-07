import copy
import os.path as osp

import argparse
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import WILLOWObjectClass as WILLOW
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
from torch_geometric.data import DataLoader
from torch_geometric.nn import SplineConv, GINConv, GCNConv, GATConv, SAGEConv, EdgeConv,GATConv, BatchNorm
from dgmc.utils import ValidPairDataset, PairDataset
from dgmc.models import DGMC, SplineCNN
import torch.nn as nn
from torch.nn import Linear as Lin
import torch.nn.functional as F
from torch_geometric.nn.pool import EdgePooling as edge_pool
from dgmc.models import MLP  # Assuming MLP is from your earlier files

from timeit import default_timer as timer

import pickle 
import numpy as np 

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--pre_epochs', type=int, default=15)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--runs', type=int, default=20)
parser.add_argument('--test_samples', type=int, default=100)
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

pre_filter1 = lambda d: d.num_nodes > 0  # noqa
pre_filter2 = lambda d: d.num_nodes > 0 and d.name[:4] != '2007'  # noqa
args.isotropic = True
print('iso: ', args.isotropic)
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

path = osp.join('..', 'data', 'PascalVOC-WILLOW')
pretrain_datasets = []
for category in PascalVOC.categories:
    dataset = PascalVOC(
        path, category, device='cpu', train=True, transform=transform, pre_filter=pre_filter2
        if category in ['car', 'motorbike'] else pre_filter1)
    pretrain_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
pretrain_dataset = torch.utils.data.ConcatDataset(pretrain_datasets)
pretrain_loader = DataLoader(pretrain_dataset, args.batch_size, shuffle=True,
                             follow_batch=['x_s', 'x_t'])

path = osp.join('..', 'data', 'WILLOW')
datasets = [WILLOW(path, cat, transform) for cat in WILLOW.categories]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# psi_1 = SplineCNN(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False,
#                   dropout=0.5)
# psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
#                   args.num_layers, cat=True, dropout=0.0)

psi_1 = HybridGraphModel(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
print('psi_1: ',psi_1)
psi_2 = HybridGraphModel(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0)
print('psi_2: ',psi_2)                  


# psi_1 = HybridGNN(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False, dropout=0.5)
# print('psi_1: ',psi_1)
# psi_2 = HybridGNN(args.rnd_dim, args.rnd_dim,  dataset.num_edge_features, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)
# psi_1 = GraphSAGE(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False, dropout=0.5)
# print('psi_1: ',psi_1)
# psi_2 = GraphSAGE(args.rnd_dim, args.rnd_dim,  dataset.num_edge_features, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)

# psi_1 = EdgeGNN(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False, dropout=0.5)
# print('psi_1: ',psi_1)
# psi_2 = EdgeGNN(args.rnd_dim, args.rnd_dim,  dataset.num_edge_features, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)

model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_voc_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def pretrain():
    model.train()

    total_loss = 0
    for data in pretrain_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_voc_y(data.y)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(pretrain_loader.dataset)


print('Pretraining model on PascalVOC...')
for epoch in range(1, args.pre_epochs + 1):
    loss = pretrain()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
state_dict = copy.deepcopy(model.state_dict())
print('Done!')


def generate_y(num_nodes, batch_size):
    row = torch.arange(num_nodes * batch_size, device=device)
    col = row[:num_nodes].view(1, -1).repeat(batch_size, 1).view(-1)
    return torch.stack([row, col], dim=0)


def train(train_loader, optimizer):
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        num_graphs = data.x_s_batch.max().item() + 1
        y = generate_y(num_nodes=10, batch_size=num_graphs)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * num_graphs

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(test_dataset):
    model.eval()

    test_loader1 = DataLoader(test_dataset, args.batch_size, shuffle=True)
    test_loader2 = DataLoader(test_dataset, args.batch_size, shuffle=True)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    correct = num_examples = 0
    times = []

    while (num_examples < args.test_samples):
        for data_s, data_t in zip(test_loader1, test_loader2):
            data_s, data_t = data_s.to(device), data_t.to(device)

            start.record()
            _, S_L = model(data_s.x, data_s.edge_index, data_s.edge_attr,
                           data_s.batch, data_t.x, data_t.edge_index,
                           data_t.edge_attr, data_t.batch)


            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

            y = generate_y(num_nodes=10, batch_size=data_t.num_graphs)

            n_correct = model.acc(S_L, y, reduction='sum')
            correct += n_correct
            num_examples += y.size(1)
            times.append(start.elapsed_time(end))

            names = [data_s.name, data_t.name]
            pos = [data_s.pos, data_t.pos]

            if num_examples >= args.test_samples:
              # print("Average inference time: " + str(sum(times)/len(times))) 
              return sum(times)/len(times), correct / num_examples, S_L, y, names, pos
            # correct += model.acc(S_L, y, reduction='sum')
            # num_examples += y.size(1)

            # if num_examples >= args.test_samples:
            #     return correct / num_examples

best_acc = 0 
best_save = []

def run(i, datasets):
    datasets = [dataset.shuffle() for dataset in datasets]
    train_datasets = [dataset[:20] for dataset in datasets]
    test_datasets = [dataset[20:] for dataset in datasets]
    train_datasets = [
        PairDataset(train_dataset, train_dataset, sample=False)
        for train_dataset in train_datasets
    ]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              follow_batch=['x_s', 'x_t'])

    model.load_state_dict(state_dict)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, 1 + args.epochs):
        start = timer()
        train(train_loader, optimizer)
        end = timer()
        time = end-start


    accs = [] 
    times = [] 
    global best_acc 
    global best_save 

    to_save = []

    for test_dataset in test_datasets: 
      t, a, S_L, y, names, pos  = test(test_dataset)
      accs.append(100*a)
      times.append(t)
      to_save.append([S_L,y,names,pos])
      print(best_acc)

    accs += [sum(accs) / len(accs)]
    times = sum(times)/len(times)

    if accs[-1] > best_acc: 
      print(accs[-1])
      best_save = to_save 
      best_acc = accs[-1]

    print(f'Run {i:02d}:')
    print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))
    print('average inference time: ' + str(times))

    return accs 
    # accs = [100 * test(test_dataset) for test_dataset in test_datasets]

    # print(f'Run {i:02d}:')
    # print(' '.join([category.ljust(13) for category in WILLOW.categories]))
    # print(' '.join([f'{acc:.2f}'.ljust(13) for acc in accs]))

    # return accs

accs = [run(i, datasets) for i in range(1, 1 + args.runs)]
print('-' * 14 * 5)
accs, stds = torch.tensor(accs).mean(dim=0), torch.tensor(accs).std(dim=0)
print(' '.join([category.ljust(13) for category in WILLOW.categories]))
print(' '.join([f'{a:.2f} ± {s:.2f}'.ljust(13) for a, s in zip(accs, stds)]))

for i in range(len(best_save)):  
  pickle.dump(best_save[i][0], open("SL" + str(i) + ".p", "wb" ))
  pickle.dump(best_save[i][1], open("y" + str(i) + ".p", "wb" ))
  pickle.dump(best_save[i][2], open("names" + str(i) + ".p", "wb" ))
  pickle.dump(best_save[i][3], open("pos" + str(i) + ".p", "wb" ))
# accs = [run(i, datasets) for i in range(1, 1 + args.runs)]
# print('-' * 14 * 5)
# accs, stds = torch.tensor(accs).mean(dim=0), torch.tensor(accs).std(dim=0)
# print(' '.join([category.ljust(13) for category in WILLOW.categories]))
# print(' '.join([f'{a:.2f} ± {s:.2f}'.ljust(13) for a, s in zip(accs, stds)]))

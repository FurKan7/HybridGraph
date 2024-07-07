import os.path as osp

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.nn import Linear as Lin
import torch.nn.functional as F

from dgmc.utils import ValidPairDataset
from dgmc.models import DGMC, SplineCNN
from torch_geometric.nn import SplineConv, GINConv, GCNConv, GATConv, SAGEConv, GMMConv, FeaStConv, EdgeConv,GATConv, BatchNorm, NNConv
from dgmc.models import MLP  # Assuming MLP is from your earlier files
from torch_geometric.nn.pool import EdgePooling as edge_pool
from timeit import default_timer as timer


parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)
args = parser.parse_args()


# import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv

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
            x = F.elu(x, alpha=1.0)
            xs.append(x)

        x = torch.cat(xs, dim=-1) if self.cat else xs[-1]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final(x) if self.lin else x
        return x
        



    def __repr__(self):
        return ('{}({}, {}, dim={}, num_layers={}, cat={}, lin={}, dropout={})').format(
            self.__class__.__name__, self.in_channels, self.out_channels, self.dim, self.num_layers, self.cat, self.lin, self.dropout)
    # def forward(self, x, edge_index, edge_attr=None):
    #     for conv in self.convs:
    #         if isinstance(conv, SplineConv):
    #             x = F.relu(conv(x, edge_index, edge_attr))
    #         else:
    #             x = F.relu(conv(x, edge_index))
    #         x = F.dropout(x, p=self.dropout, training=self.training)
    #     x = self.lin_final(x)
    #     return x



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

    

# class GATCNN(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, num_layers, heads=2, dropout=0.0):
#         super(GATCNN, self).__init__()

#         self.in_channels = in_channels  # Define in_channels as a class attribute
#         self.out_channels = out_channels  # Define out_channels as a class attribute


#         self.convs = torch.nn.ModuleList()
#         self.convs.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout))

#         for _ in range(1, num_layers - 1):
#             self.convs.append(GATConv(in_channels * heads, out_channels, heads=heads, dropout=dropout))

#         self.convs.append(GATConv(in_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))



#     def forward(self, x, edge_index, edge_attr=None):
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             x = F.relu(x)
#         return x

class GATCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, heads=1, dropout=0.0):
        super(GATCNN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels  # Setting out_channels as an instance attribute
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.residuals = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout))
        self.bns.append(BatchNorm(out_channels * heads))
        self.residuals.append(torch.nn.Linear(in_channels, out_channels * heads))

        # Intermediate layers
        for i in range(num_layers):
            if i == 0:  # First layer
                self.residuals.append(torch.nn.Linear(in_channels, out_channels * heads))
            elif i < num_layers - 1:  # Intermediate layers
                self.residuals.append(torch.nn.Linear(out_channels * heads, out_channels * heads))
            else:  # Last layer
                self.residuals.append(torch.nn.Linear(out_channels * heads, out_channels))

        # Last layer
        self.convs.append(GATConv(out_channels * heads, out_channels, heads=1, concat=False, dropout=dropout))
        self.bns.append(BatchNorm(out_channels))
        self.residuals.append(torch.nn.Identity())

    def forward(self, x, edge_index, edge_attr=None):
        for i, (conv, bn, res) in enumerate(zip(self.convs, self.bns, self.residuals)):
            identity = res(x)
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            if i < len(self.convs) - 1:  # For all but the last layer
                x = x + identity
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
# args.isotropic = True
print('iso: ', args.isotropic)
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join('..', 'data', 'PascalVOC')
for category in PascalVOC.categories:
    # print('BURADA1')
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter, device='cpu')
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    # print('BURADA2')
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'


print('num_node_features: ',dataset.num_node_features)
print('num_edge_features: ',dataset.num_edge_features)
# psi_1 = SplineCNN(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False,
#                   dropout=0.5)
# print('psi_1: ',psi_1)
# psi_2 = SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
#                   args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)

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

psi_1 = HybridGraphModel(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)
print('psi_1: ',psi_1)
psi_2 = HybridGraphModel(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0)
print('psi_2: ',psi_2)

# psi_1 = EdgeGNN(dataset.num_node_features, args.dim,
#                   dataset.num_edge_features, args.num_layers, cat=False, dropout=0.5)
# print('psi_1: ',psi_1)
# psi_2 = EdgeGNN(args.rnd_dim, args.rnd_dim,  dataset.num_edge_features, args.num_layers, cat=True, dropout=0.0)
# print('psi_2: ',psi_2)



model = DGMC(psi_1, psi_2, num_steps=args.num_steps).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


def generate_y(y_col):
    y_row = torch.arange(y_col.size(0), device=device)
    return torch.stack([y_row, y_col], dim=0)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        loss = model.loss(S_0, y)
        loss = model.loss(S_L, y) + loss if model.num_steps > 0 else loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (data.x_s_batch.max().item() + 1)

    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(dataset):
    model.eval()

    loader = DataLoader(dataset, args.batch_size, shuffle=False,
                        follow_batch=['x_s', 'x_t'])

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    correct = num_examples = 0
    times = []
    while (num_examples < args.test_samples):
        for data in loader:
            data = data.to(device)
            start.record()
            S_0, S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                             data.x_s_batch, data.x_t, data.edge_index_t,
                             data.edge_attr_t, data.x_t_batch)

            end.record()
            torch.cuda.synchronize()                 
            y = generate_y(data.y)
            correct += model.acc(S_L, y, reduction='sum')
            num_examples += y.size(1)
            times.append(start.elapsed_time(end))

            if num_examples >= args.test_samples:
                # return correct / num_examples
                return sum(times)/len(times), correct / num_examples  


for epoch in range(1, args.epochs + 1):
    start = timer()
    loss = train()
    end = timer()

    time = end-start 
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {time:.2f}')


    accs = [] 
    times = [] 

    for test_dataset in test_datasets: 
      t, a = test(test_dataset)
      accs.append(100*a)
      times.append(t)

    accs += [sum(accs) / len(accs)]
    times = sum(times)/len(times)
    
    print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
    print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))
    print('average inference time: ' + str(times))

    # accs = [100 * test(test_dataset) for test_dataset in test_datasets]
    # accs += [sum(accs) / len(accs)]

    # print(' '.join([c[:5].ljust(5) for c in PascalVOC.categories] + ['mean']))
    # print(' '.join([f'{acc:.1f}'.ljust(5) for acc in accs]))

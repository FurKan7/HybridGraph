import os.path as osp

import argparse
import torch
from torch_geometric.datasets import PascalVOCKeypoints as PascalVOC
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader

from dgmc.utils import ValidPairDataset
from dgmc.models import DGMC_modified, SplineCNN

from timeit import default_timer as timer

parser = argparse.ArgumentParser()
parser.add_argument('--isotropic', action='store_true')
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--rnd_dim', type=int, default=128)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--num_psi', type=int, default=1)
parser.add_argument('--num_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--test_samples', type=int, default=1000)
args = parser.parse_args()

pre_filter = lambda data: data.pos.size(0) > 0  # noqa
transform = T.Compose([
    T.Delaunay(),
    T.FaceToEdge(),
    T.Distance() if args.isotropic else T.Cartesian(),
])

train_datasets = []
test_datasets = []
path = osp.join('..', 'data', 'PascalVOC')
for category in PascalVOC.categories:
    dataset = PascalVOC(path, category, train=True, transform=transform,
                        pre_filter=pre_filter)
    train_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
    dataset = PascalVOC(path, category, train=False, transform=transform,
                        pre_filter=pre_filter)
    test_datasets += [ValidPairDataset(dataset, dataset, sample=True)]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)
train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                          follow_batch=['x_s', 'x_t'])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
psi_1 = SplineCNN(dataset.num_node_features, args.dim,
                  dataset.num_edge_features, args.num_layers, cat=False,
                  dropout=0.5)

psi_stack = []
for i in range(args.num_psi):
  psi_stack.append(SplineCNN(args.rnd_dim, args.rnd_dim, dataset.num_edge_features,
                  args.num_layers, cat=True, dropout=0.0))

model = DGMC_modified(psi_1, psi_stack, num_steps=args.num_steps).to(device)
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
        S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                         data.x_s_batch, data.x_t, data.edge_index_t,
                         data.edge_attr_t, data.x_t_batch)
        y = generate_y(data.y)
        loss = model.loss(S_L, y)
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
            S_L = model(data.x_s, data.edge_index_s, data.edge_attr_s,
                             data.x_s_batch, data.x_t, data.edge_index_t,
                             data.edge_attr_t, data.x_t_batch)
            end.record()
            torch.cuda.synchronize()
            y = generate_y(data.y)
            correct += model.acc(S_L, y, reduction='sum')
            num_examples += y.size(1)
            times.append(start.elapsed_time(end))

            if num_examples >= args.test_samples:
              # print("Average inference time: " + str(sum(times)/len(times))) 
              return sum(times)/len(times), correct / num_examples            

print("num psi = " + str(args.num_psi))
print("num layers = " + str(args.num_layers))
print("Isotropic = " + str(args.isotropic))

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


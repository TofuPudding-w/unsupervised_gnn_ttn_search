"""Direction-aware GIN with unordered children and fixed physical leaf labels."""
import copy
import time
import numpy as np
import torch
from torch import nn
from ttn_search.trees import features, leaves, random_tree, tree_hash, validate


def graph(tree):
    n = len(leaves(tree))
    validate(tree, n)
    x = np.zeros((2*n-1, n+5), dtype=np.float32)
    children = np.zeros((2*n-1, 2*n-1), dtype=np.float32)
    cursor = 0
    def visit(t, depth):
        nonlocal cursor
        i = cursor
        cursor += 1
        leaf = isinstance(t, int)
        x[i, 0 if i == 0 else 2 if leaf else 1] = 1
        if leaf:
            x[i, 3+t] = 1
        x[i, -2:] = depth/n, len(leaves(t))/n
        if not leaf:
            for child in t:
                j = visit(child, depth+1)
                children[i, j] = 1
        return i
    visit(tree, 0)
    return x, children


def batch(trees):
    pairs = [graph(t) for t in trees]
    return (torch.tensor(np.stack([p[0] for p in pairs])),
            torch.tensor(np.stack([p[1] for p in pairs])))


class Encoder(nn.Module):
    def __init__(self, n=8):
        super().__init__()
        self.input = nn.Linear(n+5, 64)
        self.parent = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
        self.child = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU()) for _ in range(4)])
        self.eps = nn.Parameter(torch.zeros(4))
        self.readout = nn.Sequential(nn.Linear(192,64), nn.ReLU(), nn.Linear(64,8))
        self.decoder = nn.Sequential(nn.Linear(8,64), nn.ReLU(), nn.Linear(64,n*(n-1)//2+n))

    def forward(self, x, children):
        h = torch.relu(self.input(x))
        for k in range(4):
            h = self.mlp[k]((1+self.eps[k])*h + self.child[k](children @ h)
                            + self.parent[k](children.transpose(1,2) @ h))
        root = (h*x[:, :, :1]).sum(dim=1)
        return self.readout(torch.cat([root, h.mean(dim=1), h.max(dim=1).values], dim=1))


def loss(prediction, truth, n=8):
    split = n*(n-1)//2
    return .5*((prediction[:, :split]-truth[:, :split]).square().mean()
               +(prediction[:, split:]-truth[:, split:]).square().mean())


def corpus(count, excluded, seed=20261021, n=8):
    rng = np.random.default_rng(seed)
    seen = set(excluded)
    trees = []
    while len(trees) < count:
        t = random_tree(n, rng)
        key = tree_hash(t)
        if key not in seen:
            trees.append(t)
            seen.add(key)
    return trees


def embed(model, trees):
    model.eval()
    with torch.no_grad():
        return model(*batch(trees)).numpy()


def train(trees, validation, seed, epochs=100):
    torch.manual_seed(seed)
    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    x, a = batch(trees)
    vx, va = batch(validation)
    y = torch.tensor(np.array([features(t) for t in trees]), dtype=torch.float32)
    vy = torch.tensor(np.array([features(t) for t in validation]), dtype=torch.float32)
    history = []
    best = float('inf')
    weights = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(trees))
        total = 0.
        for ids in order.split(128):
            optimizer.zero_grad()
            value = loss(model.decoder(model(x[ids], a[ids])), y[ids])
            value.backward()
            optimizer.step()
            total += float(value.detach())*len(ids)
        model.eval()
        with torch.no_grad():
            val = float(loss(model.decoder(model(vx, va)), vy))
        history.append(dict(epoch=epoch+1, train_loss=total/len(trees), validation_loss=val))
        if val < best:
            best = val
            weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(weights)
    return model, dict(seed=seed, history=history, best_validation_loss=best,
                      best_epoch=min(history, key=lambda r:r['validation_loss'])['epoch'],
                      training_time_s=time.perf_counter()-started)

"""Direction-aware GIN with unordered children and fixed physical leaf labels."""
import copy
import time
import numpy as np
import torch
from torch import nn
from ttn_search.trees import features, leaves, random_tree, tree_hash, validate


def graph(tree):
    """
    Convert a tree into a graph representation suitable for neural network processing.
    The graph representation consists of a feature matrix and an adjacency matrix.
    The feature matrix has shape (num_nodes, num_features), 
    where each row corresponds to a node in the tree
    """
    n = len(leaves(tree))
    validate(tree, n)
    # A full binary tree with n leaves has 2n-1 nodes, and each node has n+5 features:
    # 0: is_root, 1: is_internal, 2: is_leaf
    # 3..n+2: one-hot physical leaf label (only for leaves)
    # n+3: depth / n, n+4: number of leaves / n
    x = np.zeros((2*n-1, n+5), dtype=np.float32)
    # The children matrix is a binary adjacency matrix of shape (num_nodes, num_nodes)
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
        # For every node's initial feature vector(n + 5 dimensions), 
        # apply a linear transformation to project it into a 64-dimensional space.
        self.input = nn.Linear(n+5, 64)
        # 4 layers of message passing, used to capture the information from parent nodes
        self.parent = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
        # 4 layers of message passing, used to capture the information from child nodes
        self.child = nn.ModuleList([nn.Linear(64, 64, bias=False) for _ in range(4)])
        # 4 layers of nonlinear transformations after each message passing layer
        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(64,64), nn.ReLU(), nn.Linear(64,64), nn.ReLU()) for _ in range(4)])
        # A learnable parameter for each of the 4 layers
        self.eps = nn.Parameter(torch.zeros(4))
        # Aggregate the information from the root node, mean of all nodes, and max of all nodes 
        # into a 192-dimensional vector,
        self.readout = nn.Sequential(nn.Linear(192,64), nn.ReLU(), nn.Linear(64,8))
        # decoder takes the 8-dimensional vector and outputs a vector of size n*(n-1)/2 + n,
        # n(n-1)/2 for the pairwise relations and n for the leaf features.
        self.decoder = nn.Sequential(nn.Linear(8,64), nn.ReLU(), nn.Linear(64,n*(n-1)//2+n))

    def forward(self, x, children):
        # 1. The forward pass of the encoder 
        # processes the input features and the adjacency matrix of the tree.
        h = torch.relu(self.input(x))

        # 2. Pass the information through 4 layers of message passing and nonlinear transformations.
        for k in range(4):
            h = self.mlp[k]((1+self.eps[k])*h 
                            + self.child[k](children @ h)   # children @ h, collect information from child nodes
                            + self.parent[k](children.transpose(1,2) @ h))  # collect information from parent nodes
        
        # 3. Aggregate the information from the root node, mean of all nodes,
        # and max of all nodes into a 192-dimensional vector, and then apply the readout layer.
        root = (h*x[:, :, :1]).sum(dim=1)
        return self.readout(torch.cat([root, h.mean(dim=1), h.max(dim=1).values], dim=1))


def loss(prediction, truth, n=8):
    split = n*(n-1)//2
    return .5*((prediction[:, :split]-truth[:, :split]).square().mean()
               +(prediction[:, split:]-truth[:, split:]).square().mean())


def corpus(count, excluded, seed=20261021, n=8):
    """
    Construct the corpus of random trees for training and validation.
    """
    rng = np.random.default_rng(seed)
    # Exclude the trees that are already in the training set 
    # or validation set to ensure uniqueness.
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
    """
    Embed a list of trees into a feature space using the provided encoder model.
    """
    model.eval()
    with torch.no_grad():
        return model(*batch(trees)).numpy()


def train(trees, validation, seed, epochs=100):
    """
    Standard training loop for the encoder model.
    """

    # 1. Initialize the random seed for reproducibility, 
    # create an instance of the Encoder model, 
    # choose the Adam optimizer.

    torch.manual_seed(seed)
    model = Encoder()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)

    # 2. Prepare the training and validation data 
    # by batching the trees and extracting their features.

    x, a = batch(trees)
    vx, va = batch(validation)
    y = torch.tensor(np.array([features(t) for t in trees]), dtype=torch.float32)
    vy = torch.tensor(np.array([features(t) for t in validation]), dtype=torch.float32)

    # 3. Training loop
    
    history = []
    best = float('inf')
    weights = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        # Randomly shuffle the training data
        order = torch.randperm(len(trees))
        total = 0.
        # Split the training data into batches of size 128
        # and perform forward and backward passes.
        for ids in order.split(128):
            optimizer.zero_grad()
            # Forward propagation: compute the model's predictions and the loss.
            value = loss(model.decoder(model(x[ids], a[ids])), y[ids])
            # Back propagation: compute the gradients and update the model's parameters.
            value.backward()
            # update the model's parameters using the optimizer
            optimizer.step()
            total += float(value.detach())*len(ids)

        # 4. Evaluate the model on the validation set for each epoch,
        # and keep track of the best model based on validation loss.

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

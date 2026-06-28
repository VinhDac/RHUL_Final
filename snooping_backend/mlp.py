"""mlp.py - the minimal deep-learning instrument.

A small PyTorch MLP whose configuration is a few knobs we search over for the
winner's curse (search N configs, keep best-on-validation). For the gate the
searched knobs are `width` and `lr`; depth is one hidden layer, optimizer is SGD,
no regularisation (those join later - see Core.md SS6).

The maths (forward pass + the SGD update  w <- w - lr * grad,  and L2 if added
later) is derived in the dissertation; PyTorch autograd only executes it.
"""
import numpy as np
import torch
import torch.nn as nn


def make_mlp(d, width):
    """One hidden layer: d inputs -> width units (ReLU) -> 2 class logits."""
    return nn.Sequential(
        nn.Linear(d, width),
        nn.ReLU(),
        nn.Linear(width, 2),
    )


def train(X, y, width, lr, epochs=300, seed=0, return_loss=False):
    """Train an MLP on (X, y) by full-batch gradient descent (SGD).

    Returns the model, or (model, per-epoch losses) when return_loss=True.
    X: (n, d) float array.  y: (n,) int labels in {0, 1}.
    """
    torch.manual_seed(seed)                   # reproducible weights + training
    d = X.shape[1]
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    model = make_mlp(d, width)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(Xt)                    # forward pass
        loss = loss_fn(logits, yt)            # cross-entropy on the 2 logits
        loss.backward()                       # backprop: autograd computes the gradients
        optimizer.step()                      # SGD step:  w <- w - lr * grad
        if return_loss:
            losses.append(loss.item())
    return (model, losses) if return_loss else model


def accuracy(model, X, y):
    """Fraction of correct predictions of `model` on (X, y)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        pred = model(Xt).argmax(dim=1).numpy()
    return float((pred == y).mean())


def sample_config(rng):
    """Draw one random MLP config from the search space (random search).

    width ~ one of {4, 8, 16, 32, 64, 128, 256}   (capacity)
    lr    ~ log-uniform in [0.01, 1.0]            (learning rate)
    Drawing more configs = larger N; the headline sweeps N.
    """
    width = int(rng.choice([4, 8, 16, 32, 64, 128, 256]))
    lr = float(10 ** rng.uniform(-2, 0))
    return {"width": width, "lr": lr}

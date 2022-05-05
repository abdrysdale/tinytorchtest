"""Testing for torchtest"""
#! /usr/bin/env python

# Module imports
import torch

# Local imports
from torchtest import torchtest as tt
import test_networks

def test_regression():
    """Tests if a single argument regression trains"""
    torch.manual_seed(1)
    # setup test suite
    tt.setup()

    # Model
    layers = [3, 10, 1]
    model = test_networks.SingleArgRegression(layers)

    # Data
    data = [torch.rand(4, 3), torch.rand(4,1)]

    # Optimiser
    optim = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    # Loss
    loss_fn = torch.nn.MSELoss()

    # run all tests
    assert tt.test_suite(
        model,
        loss_fn,
        optim,
        data,
        test_vars_change=True,
    )

if __name__ == '__main__':
    test_regression()

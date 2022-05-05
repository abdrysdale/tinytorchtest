"""Testing for torchtest"""
#! /usr/bin/env python

# Python imports
import os
import sys

# Module imports
import torch

# Local imports
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from torchtest import torchtest as tt
import test_networks

def test_regression():
    """Tests if a single argument regression trains"""
    torch.manual_seed(1)

    # Setup test suite
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
        test_inf_vals=True,
        test_nan_vals=True,
    )

def test_regression_multi_args():
    """Tests if a multi argument regression model trains"""
    torch.manual_seed(1)

    # Setup test suite
    tt.setup()

    # Model
    layers = [3, 10, 1]
    model = test_networks.MultiArgRegression(layers)

    # Data
    data = [
        (torch.rand(4, 2), torch.rand(4,1)),    # Inputs
        torch.rand(4,1),                        # Correct outputs
    ]

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
        test_inf_vals=True,
        test_nan_vals=True,
    )

if __name__ == '__main__':
    print("Running tests...")
    test_regression()
    test_regression_multi_args()
    print("Testing complete!")
